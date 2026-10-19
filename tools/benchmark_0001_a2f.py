"""Time aligned vocal-driven A2F separately from visualization and audio muxing."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

root=Path(__file__).resolve().parent.parent
binary=root/'third_party/Audio2Face-3D-SDK/_build/local/eva-a2f-export'
audio=root/'assets/audio/vocals_v3.0.wav'
model=root/'assets/models/a2f-v3.0/model.json'
schedule=root/'outputs/vocals_Singing_claire_v3.0/emotion_schedule.json'
base=root/'outputs/0001_A2F_BENCHMARK'
base.mkdir(exist_ok=True)
alignment=json.loads((root/'outputs/0001_vocals_alignment.json').read_text())
assert alignment['reference_samples']==alignment['mixed_samples']
assert all(w['best_lag_samples']==0 and w['normalized_peak_correlation']>.5 for w in alignment['windows'])
runs=[]
for i in range(4):
    count_only=i<3
    out=base/f'inference_{i+1}' if count_only else root/'outputs/0001_Singing_claire_v3.0'
    out.mkdir(exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='8',EVA_A2F_COUNT_ONLY='1' if count_only else '0')
    command=[str(binary),str(audio),str(model),str(out),'diffusion',str(schedule)]
    begin=time.perf_counter()
    with (base/f'run_{i+1}.log').open('w') as log:
        subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    wall=time.perf_counter()-begin
    timing=json.loads((out/'timing.json').read_text())
    assert timing['frames']==6774
    if count_only:assert not list(out.glob('*.f32'))
    timing['external_process_wall_seconds']=wall
    elapsed=timing['execute_and_callback_seconds']
    timing['average_ms_per_animation_frame']=1000*elapsed/timing['frames']
    timing['animation_frames_per_second']=timing['frames']/elapsed
    timing['real_time_factor']=elapsed/timing['audio_duration_seconds']
    timing['times_faster_than_audio_duration']=timing['audio_duration_seconds']/elapsed
    runs.append(timing)
    print(json.dumps({'run':i+1,'mode':timing['mode'],'process_seconds':wall,'model_load_seconds':timing['model_load_seconds'],'execution_seconds':elapsed},ensure_ascii=False),flush=True)
keys=['model_load_seconds','cuda_context_initialization_seconds','execute_and_callback_seconds',
      'external_process_wall_seconds','average_ms_per_animation_frame',
      'animation_frames_per_second','real_time_factor','times_faster_than_audio_duration']
summary={k:{'median':statistics.median(r[k] for r in runs[:3]),
            'min':min(r[k] for r in runs[:3]),'max':max(r[k] for r in runs[:3])} for k in keys}
hashes={}
for name in ['skin.f32','tongue.f32','jaw.f32','timestamps.csv']:
    a=hashlib.sha256((root/'outputs/vocals_Singing_claire_v3.0'/name).read_bytes()).hexdigest()
    b=hashlib.sha256((root/'outputs/0001_Singing_claire_v3.0'/name).read_bytes()).hexdigest()
    hashes[name]={'matches_previous_native_output':a==b,'sha256':b}
report={'hardware':'one RTX 4090 24 GB, CUDA_VISIBLE_DEVICES=0; second GPU not used for inference benchmark',
        'model':'Audio2Face-3D v3.0 Claire, identity 0, constant noise, geometry execution All',
        'inference_audio':'aligned isolated vocals: assets/audio/vocals_v3.0.wav',
        'playback_audio':'0001.wav (vocals plus background music)',
        'audio_duration_seconds':alignment['mixed_duration_seconds'],'animation_frames':6774,
        'animation_fps':60,'benchmark_mode':'offline full-audio input, prebuilt TensorRT engine, three separate-process no-export runs plus one export run',
        'cache_note':'Each run loads its model; OS/driver caches are not cleared. This is not a reboot-cold-start measurement.',
        'scope_note':'SDK wall-clock execution with device synchronization; includes scheduling, excludes model load. Not isolated GPU kernel time.',
        'deployment_note':'Offline throughput does not establish streaming first-frame latency, P95 latency, concurrency capacity or robot response time.',
        'excluded':'TensorRT engine build, source separation, Whisper, blendshape solve, FaceMotion composition, Blender rendering, video encoding',
        'no_export_summary':summary,'runs':runs,'native_output_reproducibility':hashes}
(root/'outputs/0001_A2F_TIMING.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
