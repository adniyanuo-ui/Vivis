"""Derived voice segments shared by recognition and visual-witness collection.

No gain changes, source recording edits, identity labels or gallery writes.
Reference samples are aligned to the original PCM, never to a compressed copy.
"""
import numpy as np


FRAME=320  # 20ms at16kHz, same energy gate as speaker_observation.
PLAYBACK_RMS=.004
PLAYBACK_GUARD_FRAMES=5  # Exclude100ms before/after robot output or missing reference.


def reference_mask(pcm,far,reference_end_aligned=False):
    count=len(pcm)//2;ref=np.frombuffer(far,dtype='<i2').astype(np.float32)/32768
    missing=0
    if len(ref)!=count:
        if not reference_end_aligned:return None,'reference_alignment_unavailable'
        missing=max(0,count-len(ref))
        ref=np.r_[np.zeros(missing,np.float32),ref][-count:] if count else ref[:0]
    frames=count//FRAME
    power=np.sqrt(np.mean(ref[:frames*FRAME].reshape(-1,FRAME)**2,axis=1)) if frames else np.empty(0)
    bad=power>PLAYBACK_RMS
    bad[:min(frames,(missing+FRAME-1)//FRAME)]=True
    guarded=bad.copy()
    for offset in range(1,PLAYBACK_GUARD_FRAMES+1):
        guarded[offset:]|=bad[:-offset];guarded[:-offset]|=bad[offset:]
    return guarded,None


def clean_fragments(pcm,far,ranges,*,minimum_fragment_s=.2,padding_ms=100,reference_end_aligned=False):
    """Keep only clean, nonoverlapping source ranges, with optional visual proof."""
    from vivis.speech.speaker_observation import audio_quality
    bad,error=reference_mask(pcm,far,reference_end_aligned)
    if error:return [],error
    x=np.frombuffer(pcm,dtype='<i2').astype(np.float32)/32768
    result=[];pad=round(padding_ms*16);last_end=0
    for span in sorted(ranges,key=lambda r:r['start_s']):
        left=max(last_end,0,round(span['start_s']*16000));right=min(len(x),round(span['end_s']*16000))
        first=(left+FRAME-1)//FRAME;last=right//FRAME
        if first>=last:continue
        allowed=~bad[first:last];edges=np.diff(np.r_[False,allowed,False].astype(int))
        for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
            i=(first+int(a))*FRAME;j=(first+int(b))*FRAME
            stats={};samples,_=audio_quality(pcm[2*i:2*j],b'',minimum_fragment_s,stats)
            if samples is None:continue
            power=np.sqrt(np.mean(x[i:j].reshape(-1,FRAME)**2,axis=1));voiced=np.flatnonzero(power>.01)
            if not len(voiced):continue
            begin=max(i,i+int(voiced[0])*FRAME-pad);end=min(j,i+(int(voiced[-1])+1)*FRAME+pad)
            result.append(dict(samples=x[begin:end].copy(),effective_s=stats['effective_speech_s'],
                start_sample=begin,end_sample=end,visual_stamp=span.get('visual_stamp'),
                visual_stamps=list(span.get('visual_stamps',[])),proof=span.get('proof','confirmed_frame')))
        last_end=max(last_end,right)
    return result,None


def recognition_audio(pcm,far,*,minimum_fragment_s=.2,reference_end_aligned=False):
    """Remove robot overlap from this one utterance before any short-window merge.

    Unmodified legacy/offline calls without a reference retain their old contract.
    Live recorder calls explicitly mark end alignment: unknown reference prefixes
    are excluded, never interpreted as robot silence. Overall clipping is still
    rejected by the caller, including when it occurs in excluded audio.
    """
    if not far and not reference_end_aligned:return pcm,far,{},None
    bad,error=reference_mask(pcm,far,reference_end_aligned)
    if error:return b'',b'',{},error
    if not np.any(bad):return pcm,far,{},None
    fragments,error=clean_fragments(pcm,far,[dict(start_s=0.,end_s=len(pcm)/32000)],
        minimum_fragment_s=minimum_fragment_s,reference_end_aligned=reference_end_aligned)
    detail=dict(excluded_reference_s=float(np.count_nonzero(bad)*.02),
        clean_fragments=[dict(start_sample=f['start_sample'],end_sample=f['end_sample'],effective_s=f['effective_s']) for f in fragments])
    if error or not fragments:return b'',b'',detail,error or 'robot_playback'
    # Copy PCM bytes exactly: preserve the accepted+9dB signal and source gain.
    derived=b''.join(pcm[f['start_sample']*2:f['end_sample']*2] for f in fragments)
    return derived,bytes(len(derived)),detail,None
