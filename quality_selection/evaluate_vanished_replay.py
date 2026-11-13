"""Replay cached overlap groups once after excluding vanished blockers."""
from collections import defaultdict
from dataclasses import asdict
import argparse
import json

from analyze_miss_patterns import (
    CNN, DEST, OUT, ROOT, RANK, PQStats, assert_disjoint, cache, np, official_functions,
    parents, pd, pq_stats_from_iou, predict, read_json, rle_iou, torch, trace, write_json,
)


def replay_groups(raw, final, saved_trace, scores, protect_reopened=False):
    sources = raw["source_proposal_ids"]
    retained = {source for j in range(len(final["masks"])) for source in parents(final, j)}
    vanished = set()
    for entry in saved_trace["logs"].values():
        blockers = set(entry.get("nms_blockers", []))
        blockers.update(owner["index"] for owner in entry.get("owners", []))
        vanished.update(i for i in blockers if sources[i] not in retained)
    if not vanished:
        return final, []

    rejected = saved_trace["first_rejections"]
    active = set(range(len(sources))) - {i for row in rejected for i in row["action"]["ids"]}
    overlap = rle_iou(raw["masks"], raw["masks"]) > 0
    pending = set(vanished)
    groups = []
    affected = set()
    while pending:
        group = {min(pending)}
        frontier = set(group)
        while frontier:
            frontier = {j for i in frontier for j in active - group if overlap[i, j]}
            group.update(frontier)
        groups.append(dict(sources=[sources[i] for i in sorted(group)],
                           removed=[sources[i] for i in sorted(group & vanished)]))
        pending.difference_update(group)
        affected.update(sources[i] for i in group)

    # Disjoint raw overlap components have no shared ownership or merge edges.
    # A single image pass replays each eligible component once.
    before, _, _ = trace(raw, scores, rejected)
    assert before["final"] == final
    extra = {"action": {"kind": "reject", "ids": sorted(vanished)}}
    after, _, _ = trace(raw, scores, [*rejected, extra])
    if protect_reopened:
        nms_survivors = set(after["mask_nms"]["source_proposal_ids"])
        replay_survivors = {source for j in range(len(after["final"]["masks"]))
                            for source in parents(after["final"], j)}
        priority = set()
        for group in groups:
            eligible = []
            for source in group["sources"]:
                blockers = saved_trace["logs"][str(source)]["nms_blockers"]
                if source in nms_survivors and source not in replay_survivors and vanished.intersection(blockers):
                    eligible.append(sources.index(source))
            if eligible:
                choice = min(eligible, key=lambda i: (-scores[i], i))
                group["ownership_priority"] = sources[choice]
                priority.add(choice)
        if priority:
            after, _, _ = trace(raw, scores, [*rejected, extra], ownership_priority=priority)
    replayed = after["final"]
    def outside(row):
        return [(parents(row, j), mask, row["confidence"][j])
                for j, mask in enumerate(row["masks"]) if not affected.intersection(parents(row, j))]
    assert outside(final) == outside(replayed)
    assert_disjoint(replayed["masks"])
    return replayed, groups


def protect_clipped(raw, final, saved_trace, scores, reopened_groups):
    """Extend protection to candidates already lost at ownership in the baseline."""
    sources = raw["source_proposal_ids"]
    reopened = {source for group in reopened_groups for source in group["sources"]}
    retained = {source for j in range(len(final["masks"])) for source in parents(final, j)}
    eligible = {
        i for i, source in enumerate(sources)
        if source not in retained and source not in reopened
        and any(sources[owner["index"]] in retained
                for owner in saved_trace["logs"].get(str(source), {}).get("owners", []))
    }
    if not eligible:
        return final, []

    rejected = saved_trace["first_rejections"]
    active = set(range(len(sources))) - {i for row in rejected for i in row["action"]["ids"]}
    overlap = rle_iou(raw["masks"], raw["masks"]) > 0
    priority = set()
    affected = set()
    groups = []
    while eligible:
        choice = min(eligible, key=lambda i: (-scores[i], i))
        group = {choice}
        frontier = set(group)
        while frontier:
            frontier = {j for i in frontier for j in active - group if overlap[i, j]}
            group.update(frontier)
        eligible.difference_update(group)
        group_sources = {sources[i] for i in group}
        assert not group_sources.intersection(reopened)
        affected.update(group_sources)
        priority.add(choice)
        groups.append(dict(sources=sorted(group_sources), ownership_priority=sources[choice],
                           trigger="vanished_after_ownership"))

    removed = {source for group in reopened_groups for source in group["removed"]}
    priority.update(sources.index(group["ownership_priority"]) for group in reopened_groups
                    if "ownership_priority" in group)
    extra = {"action": {"kind": "reject", "ids": [i for i, source in enumerate(sources) if source in removed]}}
    stages, _, _ = trace(raw, scores, [*rejected, extra], ownership_priority=priority)
    replayed = stages["final"]
    def outside(row):
        return [(parents(row, j), mask, row["confidence"][j])
                for j, mask in enumerate(row["masks"]) if not affected.intersection(parents(row, j))]
    assert outside(final) == outside(replayed)
    assert_disjoint(replayed["masks"])
    return replayed, groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protect-reopened", action="store_true")
    parser.add_argument("--protect-clipped", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    raw = cache(OUT / "val")
    final = cache(CNN / "epoch6_mlp/predictions")
    saved = read_json(DEST / "traces.json")
    paths = {read_json(path)["file_name"]: path for path in (OUT / "val").glob("*.json")}
    state = torch.load(RANK / "best.pt", map_location="cpu", weights_only=False)
    predictions = {}
    decisions = {}
    for name, row in final.items():
        with np.load(paths[name].with_suffix(".npz")) as data:
            scores = predict(state, data["X"], raw[name]["confidence"], .5) if raw[name]["masks"] else []
        predictions[name], groups = replay_groups(raw[name], row, saved[name], scores,
                                                   args.protect_reopened or args.protect_clipped)
        if args.protect_clipped:
            predictions[name], extra_groups = protect_clipped(raw[name], predictions[name], saved[name], scores, groups)
            groups.extend(extra_groups)
        if groups:
            decisions[name] = groups

    records = read_json(ROOT / "local_refiner/outputs/fp_progress_three_options_20261028/validation_records.json")
    assert {rec["file_name"] for rec in records} == set(final)
    audit = read_json(DEST / "cases.json")
    cases = {(c["record"], c["target_index"]): c["id"] for c in audit}
    cohort = {c["id"] for c in audit if c["original50"]}
    totals = [PQStats(), PQStats()]
    tables = [[], []]
    changes = defaultdict(list)
    for rec in records:
        name = rec["file_name"]
        best = []
        for i, row in enumerate((final[name], predictions[name])):
            matrix = rle_iou(rec["gt"], row["masks"])
            totals[i] += pq_stats_from_iou(matrix)
            tables[i].append(dict(iou_matrix=torch.from_numpy(matrix), n_gt=len(rec["gt"]), n_pred=len(row["masks"])))
            best.append(matrix.max(axis=1, initial=0))
        for target, (old, new) in enumerate(zip(*best)):
            if (old > .5) != (new > .5):
                changes["gained" if new > .5 else "lost"].append(
                    dict(record=rec["record"], target=target, case=cases.get((rec["record"], target)),
                         before=float(old), after=float(new)))
    official = official_functions()["get_pq_score"]
    for total, table in zip(totals, tables):
        assert abs(float(official(pd.DataFrame(table))) - total.pq) < 1e-10
    assert abs(totals[0].pq - .44938154635013317) < 1e-10
    summary = dict(
        baseline=dict(**asdict(totals[0]), pq=totals[0].pq),
        replay=dict(**asdict(totals[1]), pq=totals[1].pq),
        eligible_images=len(decisions), groups=sum(map(len, decisions.values())),
        changed_images=sum(predictions[name] != row for name, row in final.items()),
        gains=len(changes["gained"]), losses=len(changes["lost"]),
        original50_gains=[item["case"] for item in changes["gained"] if item["case"] in cohort],
        passes_gate=bool(changes["gained"]) and not changes["lost"],
    )
    filename = "vanished_winner_protected.json" if args.protect_reopened else "vanished_winner_replay.json"
    if args.protect_clipped:
        filename = "expanded_ownership_protection.json"
    write_json(DEST / filename, dict(summary=summary, decisions=decisions, changes=dict(changes)))
    print(json.dumps(dict(summary=summary, changes=dict(changes)), indent=2))


if __name__ == "__main__":
    main()
