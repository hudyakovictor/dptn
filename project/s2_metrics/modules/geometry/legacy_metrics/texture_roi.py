from __future__ import annotations

import numpy as np

from .catalog_specs import specs_for_module
from .common import emit, zone_points

IMPLEMENTATION = "texture_roi.py"


def specs():
    return specs_for_module(IMPLEMENTATION, families={"F12"})


def _entropy(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0
    try:
        import cv2
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(np.clip(gray, 0, 255).astype(np.uint8)).astype(float)
    except Exception:
        pass
    hist, _ = np.histogram(gray, bins=64, range=(0, 255), density=True)
    hist = hist[hist>0]
    return float(-np.sum(hist*np.log2(hist)))


def _bbox_roi(ctx, region: str):
    if ctx.image_rgb is None or getattr(ctx.recon, "vertices_image", None) is None: return None
    img=ctx.image_rgb; h,w=img.shape[:2]
    raw=ctx.macro_indices.get(region, [])
    if not raw: return None
    idx=np.asarray(list(raw),dtype=int); vi=np.asarray(ctx.recon.vertices_image)
    idx=idx[(idx>=0)&(idx<len(vi))]
    if len(idx)<2: return None
    xy=vi[idx,:2]; xy=xy[np.isfinite(xy).all(axis=1)]
    if len(xy)<2: return None
    x0,y0=np.floor(np.min(xy,axis=0)).astype(int); x1,y1=np.ceil(np.max(xy,axis=0)).astype(int)
    pad=max(2,int(.15*max(x1-x0,y1-y0,1))); x0=max(0,x0-pad); y0=max(0,y0-pad); x1=min(w,x1+pad); y1=min(h,y1+pad)
    if x1<=x0 or y1<=y0: return None
    return img[y0:y1,x0:x1], (x1-x0)*(y1-y0)/float(w*h+1e-8)


def _poly_roi(ctx, region: str):
    if ctx.image_rgb is None or getattr(ctx.recon, "vertices_image", None) is None: return None
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None
    img=ctx.image_rgb; h,w=img.shape[:2]
    raw=ctx.macro_indices.get(region, [])
    if not raw: return None
    idx=np.asarray(list(raw),dtype=int); vi=np.asarray(ctx.recon.vertices_image)
    idx=idx[(idx>=0)&(idx<len(vi))]
    if len(idx)<3: return None
    xy=vi[idx,:2]; xy=xy[np.isfinite(xy).all(axis=1)]
    if len(xy)<3: return None
    c=np.mean(xy,axis=0); order=np.argsort(np.arctan2(xy[:,1]-c[1],xy[:,0]-c[0])); poly=xy[order]
    poly[:,0]=np.clip(poly[:,0],0,w-1); poly[:,1]=np.clip(poly[:,1],0,h-1)
    mask=Image.new('L',(w,h),0); ImageDraw.Draw(mask).polygon([tuple(map(float,p)) for p in poly], outline=1, fill=1)
    m=np.asarray(mask,dtype=bool)
    if not np.any(m): return None
    return img[m].reshape(-1,1,3), float(np.mean(m))


def _img_stats(arr, frac, prefix):
    gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(float)
    try:
        import cv2
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(np.clip(gray, 0, 255).astype(np.uint8)).astype(float)
    except Exception:
        pass
    return {
        f"{prefix}_rgb_mean": float(np.mean(arr))/255.0,
        f"{prefix}_rgb_std": float(np.std(arr))/255.0,
        f"{prefix}_gray_entropy": _entropy(gray),
        f"{prefix}_micro_contrast": float(np.std(gray)/(np.mean(gray)+1e-8)),
        f"{prefix}_area_fraction": float(frac),
    }


def compute(ctx, specs_):
    out=[]; spec_by={s.name:s for s in specs_}; vals={}
    regions=set()
    for spec in specs_:
        if spec.name.startswith('roi_'):
            parts=spec.name.split('_')
            # recover known region between roi_ and metric suffix
            for r in ctx.macro_indices:
                if spec.name.startswith('roi_'+r+'_'):
                    regions.add(r)
    for region in regions:
        bbox=_bbox_roi(ctx,region)
        if bbox is not None:
            arr,frac=bbox; st=_img_stats(arr,frac,f"roi_{region}")
            # rename area key to bbox_area_fraction
            if f"roi_{region}_area_fraction" in st:
                st[f"roi_{region}_bbox_area_fraction"]=st.pop(f"roi_{region}_area_fraction")
            vals.update(st)
        poly=_poly_roi(ctx,region)
        if poly is not None:
            arr,frac=poly; st=_img_stats(arr,frac,f"roi_{region}_poly")
            vals.update(st)
        elif bbox is not None:
            arr,frac=bbox; st=_img_stats(arr,frac,f"roi_{region}_poly")
            vals.update(st)
        pts=zone_points(ctx,region)
        if len(pts)>1:
            raw = ctx.macro_indices.get(region, [])
            uv_vals = None
            if ctx.uv_coords is not None and raw:
                try:
                    idx=np.asarray(list(raw),dtype=int)
                    uv=np.asarray(ctx.uv_coords)
                    idx=idx[(idx>=0)&(idx<len(uv))]
                    if len(idx)>1:
                        uvr=uv[idx,:2]
                        uv_vals={
                            "uv_mean": float(np.mean(uvr)),
                            "uv_p95": float(np.percentile(np.linalg.norm(uvr-np.mean(uvr,axis=0),axis=1),95)),
                            "uv_span": float(np.linalg.norm(np.ptp(uvr,axis=0))),
                        }
                except Exception:
                    uv_vals=None
            vals[f"roi_{region}_canon_depth_mean"] = float(np.mean(pts[:, 2]))
            vals[f"roi_{region}_canon_depth_p95"] = float(np.percentile(pts[:, 2], 95))
            if uv_vals is not None:
                vals[f"roi_{region}_uv_coord_mean"] = uv_vals["uv_mean"]
                vals[f"roi_{region}_uv_coord_p95"] = uv_vals["uv_p95"]
                vals[f"roi_{region}_uv_coord_span"] = uv_vals["uv_span"]
            normals = ctx.normals_canon
            if normals is not None and raw:
                idx=np.asarray(list(raw),dtype=int); idx=idx[(idx>=0)&(idx<len(normals))]
                if len(idx)>1:
                    ns=np.asarray(normals[idx]); vals[f"roi_{region}_uv_normal_entropy"] = float(np.mean(np.var(ns,axis=0)))
                else:
                    vals[f"roi_{region}_uv_normal_entropy"] = float(np.std(pts[:,2])/(abs(np.mean(pts[:,2]))+1e-8))
            else:
                vals[f"roi_{region}_uv_normal_entropy"] = float(np.std(pts[:,2])/(abs(np.mean(pts[:,2]))+1e-8))
            vals[f"roi_{region}_uv_depth_histogram_pca1"] = float(np.ptp(pts[:,2])) if uv_vals is None else uv_vals["uv_span"]
    # ROI deltas from computed stats.
    for side in ('L','R'):
        o=vals.get(f"roi_orbit_{side}_poly_gray_entropy", vals.get(f"roi_orbit_{side}_gray_entropy"))
        c=vals.get(f"roi_cheekbone_{side}_poly_gray_entropy", vals.get(f"roi_cheekbone_{side}_gray_entropy"))
        if o is not None and c is not None: vals[f"texture_periocular_minus_cheek_{side}"]=o-c
        b=vals.get(f"roi_brow_ridge_{side}_poly_micro_contrast", vals.get(f"roi_brow_ridge_{side}_micro_contrast"))
        if b is not None and o is not None: vals[f"brow_lid_texture_delta_{side}"]=b-o
        t=vals.get(f"roi_temporal_{side}_poly_micro_contrast", vals.get(f"roi_temporal_{side}_micro_contrast"))
        if t is not None and c is not None: vals[f"temporal_edge_texture_delta_{side}"]=t-c
    nb=vals.get('roi_nose_bridge_tip_poly_micro_contrast', vals.get('roi_nose_bridge_tip_micro_contrast'))
    if nb is not None: vals['bridge_texture_delta']=nb
    vals['roi_mask_boundary_discontinuity']=max([abs(v) for k,v in vals.items() if 'texture_delta' in k or 'minus_cheek' in k] or [0.0])
    vals['roi_silicone_transition_score']=vals['roi_mask_boundary_discontinuity']
    for name,val in vals.items():
        spec=spec_by.get(name)
        if spec and (mv:=emit(spec,val,confidence=0.58)): out.append(mv)
    return out
