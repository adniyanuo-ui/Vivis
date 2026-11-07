"""Optional local Chinese TTS for cached menu prompts, never device playback."""
import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
import wave
import numpy as np
from vivis.paths import ROOT

PROCESSING_VERSION='r128-compand-v1'

TEXTS={
    'welcome':'你好。可以说，播放白骨夫人，或者，开始表情跟随。',
    'menu':'目前有白骨夫人和表情跟随两个节目。你想选哪一个？',
    'clarify':'请说，播放白骨夫人，或者，开始表情跟随。',
    'standby':'好的，我先休息一会儿。',
    'fault':'运行遇到问题，已经停止动作。请查看日志。',
    'identity':'我是小帅，可以为你表演白骨夫人，也可以模仿你的表情。',
    'stop_test':'停止。结束。退出。不用学了。待机。休息一下。自己玩一会儿。',
    'select_test':'播放白骨夫人。开始表情跟随。',
    'test_stop':'停止。','test_end':'结束。','test_exit':'退出。',
    'test_idle':'休息一下。','test_selfplay':'自己玩一会儿。',
    'test_mirror':'开始表情跟随。','test_song':'播放白骨夫人。'
}


def cache_matches(config,metadata):
    return (isinstance(metadata,dict) and metadata.get('processing_version')==PROCESSING_VERSION and metadata.get('tts')==config['tts']
        and metadata.get('level',{}).get('config')==config.get('prompt_level') and metadata.get('texts')==TEXTS)


def read_pcm16(path):
    with wave.open(str(path)) as wav:
        if wav.getsampwidth()!=2:raise ValueError('Prompt reference must be PCM16 WAV')
        rate=wav.getframerate()
        data=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').reshape(-1,wav.getnchannels()).astype(np.float64)/32768
        if data.shape[1]==1:data=data[:,0]
    return data,rate


def measured_loudness(data,rate):
    import pyloudnorm as pyln
    if not len(data) or not np.isfinite(data).all():raise ValueError('Invalid prompt samples')
    if len(data)<math.ceil(rate*.4):
        data=np.pad(data,((0,math.ceil(rate*.4)-len(data)),(0,0)) if data.ndim==2 else (0,math.ceil(rate*.4)-len(data)))
    value=float(pyln.Meter(rate).integrated_loudness(data))
    if not math.isfinite(value):raise ValueError('No measurable speech loudness')
    return value


def normalize_prompt(path,reference_lufs,settings):
    """Offline R128 gain; compress only if the requested gain would exceed peaks."""
    from scipy.signal import resample_poly
    offset=settings['relative_loudness_lu'];ceiling=settings['true_peak_dbfs']
    threshold=settings['compression_threshold_dbfs'];ratio=settings['compression_ratio']
    if (any(type(v) not in (int,float) or not math.isfinite(v) for v in (offset,ceiling,threshold,ratio))
            or not -12<=offset<0 or not -9<=ceiling<=-1 or not -30<=threshold<=-6 or not 1<=ratio<=6):
        raise ValueError('Invalid prompt level configuration')
    data,rate=read_pcm16(path);before=measured_loudness(data,rate);target=reference_lufs+offset
    peak=lambda x:float(np.max(np.abs(resample_poly(x,4,1,axis=0))))
    limit=10**(ceiling/20);compressed=False
    if peak(data)*10**((target-before)/20)>limit:
        if shutil.which('sox') is None:raise RuntimeError('Prompt preparation requires: sudo apt-get install sox')
        processed=path.with_suffix('.companded.wav')
        try:
            subprocess.run(['sox',str(path),str(processed),'compand','0,0.08',
                f'-80,-80,{threshold},{threshold},0,{threshold-threshold/ratio}','0','-90','0.005'],
                check=True,capture_output=True,text=True,timeout=60)
            data,rate=read_pcm16(processed);compressed=True
        finally:
            if processed.exists():processed.unlink()
    current=measured_loudness(data,rate)
    gain=min(10**((target-current)/20),limit/peak(data));data=data*gain
    measured=measured_loudness(data,rate)
    return data,dict(raw_lufs=before,target_lufs=target,measured_lufs=measured,relative_lu=measured-reference_lufs,
        true_peak_dbfs=20*math.log10(peak(data)),gain=gain,compressed=compressed)


def prepare(config):
    import sherpa_onnx
    t=config['tts'];folder=ROOT/t['directory'];destination=ROOT/config['prompts'];destination.mkdir(parents=True,exist_ok=True)
    metadata=dict(processing_version=PROCESSING_VERSION,tts=t,texts=TEXTS,asset_sha256={k:hashlib.sha256((folder/t[k]).read_bytes()).hexdigest()
        for k in ('model','tokens','voices','lexicon') if k in t})
    level=config.get('prompt_level');reference_channels=1
    if level:
        reference=ROOT/level['reference_audio'];data,rate=read_pcm16(reference)
        if level.get('method')!='r128':raise ValueError('Unsupported prompt normalization')
        reference_channels=data.shape[1] if data.ndim==2 else 1
        reference_loudness=measured_loudness(data,rate)
        metadata['level']=dict(config=level,reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                               reference_lufs=reference_loudness,reference_channels=reference_channels)
    try:cached=json.loads((destination/'metadata.json').read_text())
    except (OSError,ValueError):cached={}
    if not isinstance(cached,dict):cached={}
    common=dict(model=str(folder/t['model']),tokens=str(folder/t['tokens']),lexicon=str(folder/t['lexicon']))
    if t.get('engine','vits')=='kokoro':
        engine_model=dict(kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(**common,voices=str(folder/t['voices'])))
    elif t.get('engine','vits')=='vits':
        engine_model=dict(vits=sherpa_onnx.OfflineTtsVitsModelConfig(**common,dict_dir=str(folder/t['dict'])))
    else:raise ValueError('Unsupported local TTS engine')
    model=sherpa_onnx.OfflineTtsModelConfig(**engine_model,num_threads=t['threads'],provider='cpu')
    rules=','.join(str(folder/n) for n in t.get('rule_fsts',('date.fst','number.fst','phone.fst')))
    engine=sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=model,rule_fsts=rules))
    sid=t.get('speaker_id',0);speed=t.get('speed',1.)
    if type(sid) is not int or not 0<=sid<engine.num_speakers or not .5<=speed<=2:raise ValueError('Invalid prompt speaker or speed')
    reports={}
    for name,text in TEXTS.items():
        path=destination/(name+'.wav')
        if (path.is_file() and all(cached.get(k)==v for k,v in metadata.items() if k!='texts')
                and cached.get('texts',{}).get(name)==text):
            reports[name]=cached.get('level_reports',{}).get(name,{})
            continue
        audio=engine.generate(text,sid=sid,speed=speed);samples=np.asarray(audio.samples)
        if reference_channels>1:samples=np.repeat(samples[:,None],reference_channels,axis=1)
        temporary=path.with_suffix('.tmp.wav')
        with wave.open(str(temporary),'wb') as wav:
            wav.setnchannels(reference_channels);wav.setsampwidth(2);wav.setframerate(audio.sample_rate)
            wav.writeframes(np.clip(samples*32767,-32768,32767).astype('<i2').tobytes())
        if level:
            samples,reports[name]=normalize_prompt(temporary,reference_loudness,level)
            with wave.open(str(temporary),'wb') as wav:
                wav.setnchannels(reference_channels);wav.setsampwidth(2);wav.setframerate(audio.sample_rate)
                wav.writeframes(np.clip(samples*32767,-32768,32767).astype('<i2').tobytes())
        temporary.replace(path)
        print('Prepared prompt: '+name,flush=True)
    (destination/'texts.json').write_text(json.dumps(TEXTS,ensure_ascii=False,indent=2)+'\n')
    metadata['level_reports']=reports
    temporary=destination/'metadata.tmp';temporary.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n');temporary.replace(destination/'metadata.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare(json.loads((ROOT/'configs/perception/speech.json').read_text()))
