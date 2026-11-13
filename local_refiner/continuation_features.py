"""Prediction/image-only diagnostics of a proposed filament continuation."""
import cv2
import numpy as np
from scipy import ndimage as ndi


def skeleton(mask):
    work=np.pad(mask.astype(np.uint8),1);result=np.zeros_like(work)
    cross=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    while work.any():
        eroded=cv2.erode(work,cross,borderType=cv2.BORDER_CONSTANT,borderValue=0)
        opened=cv2.dilate(eroded,cross,borderType=cv2.BORDER_CONSTANT,borderValue=0)
        result|=work&~opened;work=eroded
    return result[1:-1,1:-1].astype(bool)


def shape_description(mask):
    skel=skeleton(mask);distance=ndi.distance_transform_edt(mask)
    widths=2*distance[skel]
    yy,xx=np.nonzero(skel);points=np.column_stack([xx,yy]).astype(float)
    return skel,points,float(np.median(widths)) if len(widths) else None


def local_axis(points,point,radius):
    subset=points[np.linalg.norm(points-point,axis=1)<=radius]
    if len(subset)<4:return None,None,len(subset)
    cov=np.cov(subset.T);values,vectors=np.linalg.eigh(cov)
    ratio=float(values[-1]/max(values[0],1e-6))
    return vectors[:,-1],ratio,len(subset)


def measure_continuation(anchor,added,image):
    """Masks and grayscale image share a crop. No annotations are accepted."""
    anchor=anchor.astype(bool);added=added.astype(bool)
    if anchor.shape!=added.shape or image.shape!=anchor.shape or not anchor.any() or not added.any():
        raise ValueError('Continuation needs two nonempty masks and an aligned grayscale image')
    if (anchor&added).any():raise ValueError('Use disjoint final masks for continuation features')
    def main_component(mask):
        labels,n=ndi.label(mask,np.ones((3,3),int))
        sizes=np.bincount(labels.ravel());sizes[0]=0
        return labels==int(sizes.argmax())
    amain=main_component(anchor);bmain=main_component(added)
    askel,apoints,awidth=shape_description(amain)
    bskel,bpoints,bwidth=shape_description(bmain)
    # Choose the anchor skeleton endpoint nearest the added region; fall back
    # to any skeleton point when morphology produces no simple endpoint.
    neighbours=cv2.filter2D(askel.astype(np.uint8),cv2.CV_16S,np.ones((3,3),np.int16))
    endpoints=askel&(neighbours==2)
    ey,ex=np.nonzero(endpoints if endpoints.any() else askel)
    distance_to_added=ndi.distance_transform_edt(~bmain)
    j=int(np.argmin(distance_to_added[ey,ex]));aend=np.array([ex[j],ey[j]],float)
    bend=bpoints[int(np.argmin(np.linalg.norm(bpoints-aend,axis=1)))]
    aradius=float(np.clip(3*awidth,10,80));bradius=float(np.clip(3*bwidth,10,80))
    aa,aratio,an=local_axis(apoints,aend,aradius);ba,bratio,bn=local_axis(bpoints,bend,bradius)
    angle=None if aa is None or ba is None else float(np.degrees(np.arccos(np.clip(abs(float(aa@ba)),0,1))))
    width_ratio=float(max(awidth,bwidth)/max(min(awidth,bwidth),1e-6))
    # Local brightness reference, fixed before seeing annotation outcomes.
    intens=image.astype(np.float32)
    mean=cv2.GaussianBlur(intens,(0,0),12,borderType=cv2.BORDER_REFLECT)
    variance=np.maximum(cv2.GaussianBlur(intens*intens,(0,0),12,borderType=cv2.BORDER_REFLECT)-mean*mean,0)
    darkness=(mean-intens)/(np.sqrt(variance)+2.)
    valid=intens>10
    def intensity_stats(mask):
        vals=darkness[mask&valid]
        return dict(median_local_darkness=float(np.median(vals)) if len(vals) else None,
                    dark_fraction=float((vals>.25).mean()) if len(vals) else None)
    aimage=intensity_stats(anchor);bimage=intensity_stats(added)
    return dict(tangent_angle_degrees=angle,anchor_axis_ratio=aratio,added_axis_ratio=bratio,
        anchor_local_points=an,added_local_points=bn,anchor_width=awidth,added_width=bwidth,width_ratio=width_ratio,
        contact_gap_pixels=float(distance_to_added[ey[j],ex[j]]),anchor_endpoint=aend.tolist(),added_join=bend.tolist(),
        anchor_local_axis=None if aa is None else aa.tolist(),added_local_axis=None if ba is None else ba.tolist(),
        added_area_ratio=float(added.sum()/anchor.sum()),anchor_image=aimage,added_image=bimage,
        darkness_difference=None if aimage['median_local_darkness'] is None or bimage['median_local_darkness'] is None else bimage['median_local_darkness']-aimage['median_local_darkness'])
