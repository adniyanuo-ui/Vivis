"""Opt-in named channel scaling before the existing calibration mapper."""
import hashlib
import numpy as np

from vivis.motion.mapping import inputs,compile_trajectory


def round_lip_gain_targets(robot,work,ids,positions,fps,gain=1.):
    """Recompile Funnel/Pucker only; preserve accepted unrelated target columns."""
    if type(gain) not in (int,float) or not np.isfinite(gain) or gain<=0:
        raise ValueError('Round-lip channel gain must be finite and positive')
    result=np.array(positions,dtype=float,copy=True)
    report=dict(gain=float(gain),channels=['MouthFunnel','MouthPucker'],affected_ids=[],changed_ids=[])
    if gain==1.:return result,report
    source=work.get('source_clip')
    if source is None or not work.get('source_clip_sha256'):
        raise ValueError('This work has no verified source clip for runtime channel gain')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if digest!=work['source_clip_sha256']:raise ValueError('Channel gain source SHA256 mismatch')
    values,source_fps,order,shapes=inputs(source,robot.calibration,robot.resolve('channels'))
    if source_fps==30 and fps==60:values=np.repeat(values,2,axis=0)
    elif source_fps!=fps:raise ValueError('Channel gain requires matching FPS or30-to60 held samples')
    if result.shape!=(len(values),len(ids)):raise ValueError('Channel gain source timing/shape mismatch')
    profile=robot.motion_profile()
    mapped_ids,baseline,_=compile_trajectory(values,order,shapes,profile,'face_eyes')
    if mapped_ids!=ids:raise ValueError('Channel gain source motor group differs from playback')
    affected=sorted({sid for name in report['channels'] for sid in shapes[name]})
    columns=[ids.index(sid) for sid in affected]
    error=float(np.max(np.abs(baseline[:,columns]-result[:,columns])))
    if error>1e-6:
        raise ValueError('Channel gain source does not reproduce accepted lip targets; review provenance')
    report.update(source_sha256=digest,source_fps=source_fps,playback_fps=fps,
                  affected_ids=affected,baseline_reproduction_max_error_counts=error,coefficients={})
    for name in report['channels']:
        column=order.index(name);before=values[:,column].copy();scaled=before*gain
        if not np.isfinite(scaled).all():raise ValueError('Round-lip channel gain overflow')
        values[:,column]=np.clip(scaled,0,1)
        report['coefficients'][name]=dict(peak_before=float(before.max()),peak_after=float(values[:,column].max()),
                                        saturated_frames=int(np.count_nonzero(scaled>1)))
    _,mapped,mapping_report=compile_trajectory(values,order,shapes,profile,'face_eyes')
    result[:,columns]=mapped[:,columns]
    report['changed_ids']=[sid for sid,j in zip(ids,range(len(ids))) if np.any(result[:,j]!=positions[:,j])]
    report['position_clipped_frames_by_id']={sid:mapping_report['clipped_samples_by_id'][str(sid)] for sid in affected}
    return result,report
