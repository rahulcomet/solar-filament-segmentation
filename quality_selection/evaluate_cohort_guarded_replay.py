"""Audit and evaluate preservation-gated alternatives for vanished masks."""
from collections import defaultdict
from dataclasses import asdict
from scipy.optimize import linear_sum_assignment

from evaluate_vanished_replay import (
    CNN, DEST, OUT, ROOT, RANK, PQStats, assert_disjoint, cache, np, official_functions,
    parents, pd, pq_stats_from_iou, predict, read_json, replay_groups, rle_iou, torch, trace, write_json,
)
from group_actions import candidates


def preservation(before, after):
    if not before:
        return 1.0
    if len(after) < len(before):
        return 0.0
    overlap = rle_iou(before, after)
    # Prefer assignments that preserve the greatest number of old instances.
    rows, cols = linear_sum_assignment((overlap >= .8).astype(float) * (len(before) + 1) + overlap,
                                       maximize=True)
    return float(overlap[rows, cols].min())


def score(records, predictions):
    total = PQStats()
    table = []
    hits = {}
    for rec in records:
        masks = predictions[rec["file_name"]]
        matrix = rle_iou(rec["gt"], masks)
        total += pq_stats_from_iou(matrix)
        table.append(dict(iou_matrix=torch.from_numpy(matrix), n_gt=len(rec["gt"]), n_pred=len(masks)))
        hits[rec["record"]] = matrix.max(axis=1, initial=0)
    assert abs(float(official_functions()["get_pq_score"](pd.DataFrame(table))) - total.pq) < 1e-10
    return dict(**asdict(total), pq=total.pq), hits


def main():
    torch.set_num_threads(2)
    raw = cache(OUT / "val")
    final = cache(CNN / "epoch6_mlp/predictions")
    saved = read_json(DEST / "traces.json")
    records = read_json(ROOT / "local_refiner/outputs/fp_progress_three_options_20261028/validation_records.json")
    by_image = defaultdict(list)
    for rec in records:
        by_image[rec["file_name"]].append(rec)
    audit = [c for c in read_json(DEST / "cases.json") if c["original50"]]
    assert len(audit) == 48
    by_case_image = defaultdict(list)
    for case in audit:
        by_case_image[case["file_name"]].append(case)
    paths = {read_json(path)["file_name"]: path for path in (OUT / "val").glob("*.json")}
    state = torch.load(RANK / "best.pt", map_location="cpu", weights_only=False)
    controls, outputs, decisions, findings = {}, {}, {}, []
    for number, (name, row) in enumerate(raw.items(), 1):
        source_ids = row["source_proposal_ids"]
        with np.load(paths[name].with_suffix(".npz")) as data:
            scores = predict(state, data["X"], row["confidence"], .5) if source_ids else np.zeros(0)
        control, reopened = replay_groups(row, final[name], saved[name], scores, True)
        controls[name] = control["masks"]
        outputs[name] = list(control["masks"])
        reserved = {s for group in reopened for s in group["sources"]}
        retained = {s for j in range(len(control["masks"])) for s in parents(control, j)}
        rejected = {i for action in saved[name]["first_rejections"] for i in action["action"]["ids"]}
        active = [i for i in range(len(source_ids)) if i not in rejected]
        active_row = dict(file_name=name, **{k: [row[k][i] for i in active]
                          for k in ("masks", "confidence", "source_proposal_ids")})
        components, _ = candidates(active_row)
        accepted = []
        pending_cases = {c["id"]: c for c in by_case_image[name]}
        for component in components:
            members = [active[i] for i in component]
            sources = {source_ids[i] for i in members}
            original = [j for j in range(len(control["masks"])) if sources.intersection(parents(control, j))]
            before = [control["masks"][j] for j in original]
            eligible = []
            for i in members:
                s = source_ids[i]
                entry = saved[name]["logs"].get(str(s), {})
                if s not in retained and (entry.get("nms_blockers") or entry.get("owners")):
                    eligible.append(i)
            # Audit every qualifying source, including sources in the already protected groups.
            audited = [c for c in pending_cases.values() if c["raw_source"] in sources]
            for c in audited:
                pending_cases.pop(c["id"])
            if not eligible and not audited:
                continue
            group_row = dict(file_name=name, **{k: [row[k][i] for i in members]
                             for k in ("masks", "confidence", "source_proposal_ids")})
            group_scores = np.asarray(scores)[members]
            alternatives = {}
            audit_indices = {source_ids.index(c["raw_source"]) for c in audited}
            for i in sorted(set(eligible) | audit_indices):
                local = members.index(i)
                promoted = group_scores.copy()
                promoted[local] = max(group_scores) + 1
                stages, _, _ = trace(group_row, promoted, [], ownership_priority={local})
                after = stages["final"]["masks"]
                guard = preservation(before, after)
                alternatives[i] = (guard, after)
            for c in audited:
                target = next(r["gt"][c["target_index"]] for r in by_image[name] if r["record"] == c["record"])
                i = source_ids.index(c["raw_source"])
                guard, after = alternatives[i]
                other = [m for j, m in enumerate(control["masks"]) if j not in original]
                gains = losses = 0
                for rec in by_image[name]:
                    old = (rle_iou(rec["gt"], control["masks"]) > .5).any(axis=1)
                    new = (rle_iou(rec["gt"], other + after) > .5).any(axis=1)
                    gains += int((new & ~old).sum())
                    losses += int((old & ~new).sum())
                findings.append(dict(id=c["id"], first_loss=c["first_loss"],
                    direction="contained" if c["contained_by_larger"] else "extended" if c["broader_than_blocker"] else "other",
                    control_iou=float(rle_iou([target], control["masks"]).max(initial=0)),
                    promoted_iou=float(rle_iou([target], other + after).max(initial=0)),
                    preservation=guard, image_gains=gains, image_losses=losses))
            if sources.intersection(reserved):
                continue
            safe = [i for i in eligible if alternatives[i][0] >= .8 and len(alternatives[i][1]) > len(before)]
            if safe:
                choice = min(safe, key=lambda i: (-alternatives[i][0], -scores[i], i))
                accepted.append((original, alternatives[choice][1]))
                decisions.setdefault(name, []).append(dict(source=source_ids[choice], preservation=alternatives[choice][0]))
        for c in pending_cases.values():
            findings.append(dict(id=c["id"], first_loss=c["first_loss"], direction="other",
                                 unavailable="source is outside active overlap groups"))
        if accepted:
            replaced = {i for old, _ in accepted for i in old}
            outputs[name] = [m for i, m in enumerate(control["masks"]) if i not in replaced]
            outputs[name].extend(m for _, masks in accepted for m in masks)
            assert_disjoint(outputs[name])
        if number % 25 == 0:
            print(f"Processed {number}/{len(raw)} images", flush=True)
    assert len(findings) == 48 and len({row["id"] for row in findings}) == 48
    baseline, before = score(records, controls)
    result, after = score(records, outputs)
    assert abs(baseline["pq"] - .4502066695297027) < 1e-10
    changes = dict(gained=[], lost=[])
    for record, old in before.items():
        for i, (a, b) in enumerate(zip(old, after[record])):
            if (a > .5) != (b > .5):
                changes["gained" if b > .5 else "lost"].append(
                    dict(record=record, target=i, before=float(a), after=float(b)))
    summary = dict(control=baseline, result=result, changed_images=len(decisions),
                   gained=len(changes["gained"]), lost=len(changes["lost"]))
    write_json(DEST / "cohort_guarded_replay.json",
               dict(summary=summary, audit=findings, decisions=decisions, changes=changes))
    print(summary, flush=True)


if __name__ == "__main__":
    main()
