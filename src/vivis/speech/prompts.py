"""Optional local Chinese TTS for cached menu prompts, never device playback."""
import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import os
import sys
from pathlib import Path
import wave
import numpy as np
from vivis.paths import ROOT

PROCESSING_VERSION='r128-compand-pauses-v2'
RUNTIME_PROMPTS=('welcome','menu','clarify','standby','interaction','identity','fault')

TEXTS={
    'welcome':'在呢，你说！',
    'menu':'我能唱白骨夫人、演魏总剧本，还能学你的表情。你点节目，我来演！',
    'clarify':'你可以说，播放白骨夫人，播放魏总剧本，或者开始表情跟随。',
    'standby':'好嘞，我歇会儿。有事喊小帅！',
    'interaction':'好，咱们接着聊。',
    'fault':'刚才出了点问题，我先停下来。请检查设备，再重新启动。',
    'identity':'我是小帅，二十岁，北京男孩。唱歌、演戏、学表情，随你挑！',
    'stop_test':'停止。结束。退出。不用学了。待机。休息一下。自己玩一会儿。',
    'select_test':'播放白骨夫人。开始表情跟随。',
    'test_stop':'停止。','test_end':'结束。','test_exit':'退出。',
    'test_idle':'休息一下。','test_selfplay':'自己玩一会儿。',
    'test_mirror':'开始表情跟随。','test_song':'播放白骨夫人。',
    'test_wake':'小帅。','test_return':'不用播了。'
}


def tts_signature(tts):
    if not isinstance(tts,dict) or tts.get('engine')!='qwen3':return tts
    # Prepared WAVs are portable. Record the generation environment but do not
    # force customers to recreate a GPU job just to play verified cached audio.
    return {k:v for k,v in tts.items() if k not in ('python','threads','device','dtype')}


def expected_texts(config):
    overrides=config.get('prompt_text_overrides',{})
    if (not isinstance(overrides,dict) or not set(overrides)<=set(RUNTIME_PROMPTS)
            or any(not isinstance(t,str) or not 1<=len(t)<=600 for t in overrides.values())):
        raise ValueError('Invalid system prompt text overrides')
    return {**TEXTS,**overrides}


def cache_matches(config,metadata):
    return (isinstance(metadata,dict) and metadata.get('processing_version')==PROCESSING_VERSION and tts_signature(metadata.get('tts'))==tts_signature(config['tts'])
        and metadata.get('timing')==config.get('prompt_timing')
        and metadata.get('level',{}).get('config')==config.get('prompt_level')
        and isinstance(metadata.get('texts'),dict)
        and all(metadata['texts'].get(name)==expected_texts(config)[name] for name in RUNTIME_PROMPTS))


def prompt_cache_ready(config):
    """Verify small cached WAVs; runtime never needs to load the TTS model."""
    directory=ROOT/config['prompts']
    try:
        metadata=json.loads((directory/'metadata.json').read_text())
        if not cache_matches(config,metadata):return False
        manifest=config['tts'].get('source_manifest')
        if manifest and metadata.get('asset_sha256',{}).get('source_manifest')!=hashlib.sha256((ROOT/manifest).read_bytes()).hexdigest():
            return False
        for name in RUNTIME_PROMPTS:
            path=directory/(name+'.wav')
            if not path.is_file():return False
            expected=metadata.get('prompt_sha256',{}).get(name)
            if config['tts'].get('engine') in ('qwen3','dashscope') and not expected:return False
            if expected and hashlib.sha256(path.read_bytes()).hexdigest()!=expected:return False
            with wave.open(str(path)) as wav:
                if wav.getsampwidth()!=2 or wav.getnchannels() not in (1,2) or wav.getnframes()<=0:return False
        return True
    except (OSError,ValueError,TypeError,wave.Error,EOFError):return False


def synthesize_prompt(engine,text,sid,speed,timing=None):
    """Generate cached clauses with explicit silent gaps, never live playback."""
    if timing is None:
        audio=engine.generate(text,sid=sid,speed=speed)
        return np.asarray(audio.samples),audio.sample_rate,{}
    for key,low,high in (('clause_pause_ms',80,500),('sentence_pause_ms',200,1000),('edge_fade_ms',0,20)):
        value=timing.get(key)
        if type(value) not in (int,float) or not math.isfinite(value) or not low<=value<=high:
            raise ValueError('Invalid prompt timing: '+key)
    clauses=re.findall(r'[^，,。.!！？?；;]+[，,。.!！？?；;]*',text)
    parts=[];rate=None;report=[];cursor=0
    for index,clause in enumerate(clauses):
        clause=clause.strip()
        audio=engine.generate(clause,sid=sid,speed=speed)
        samples=np.array(audio.samples,dtype=float,copy=True)
        if (samples.ndim!=1 or not len(samples) or not np.isfinite(samples).all()
                or type(audio.sample_rate) is not int or audio.sample_rate<=0
                or rate is not None and rate!=audio.sample_rate):
            raise ValueError('Invalid synthesized prompt clause')
        rate=audio.sample_rate
        # Remove only outer near-silence; keep a20ms speech edge before applying
        # a short fade so concatenating independently synthesized clauses clicks less.
        active=np.flatnonzero(np.abs(samples)>max(1e-4,float(np.max(np.abs(samples)))*.01))
        if not len(active):raise ValueError('Silent synthesized prompt clause')
        margin=round(rate*.02);samples=samples[max(0,active[0]-margin):min(len(samples),active[-1]+margin+1)]
        fade=min(round(rate*timing['edge_fade_ms']/1000),len(samples)//2)
        if fade:
            envelope=np.linspace(0.,1.,fade);samples[:fade]*=envelope;samples[-fade:]*=envelope[::-1]
        parts.append(samples);cursor+=len(samples)
        if index+1<len(clauses):
            key='sentence_pause_ms' if re.search(r'[。.!！？?；;]$',clause) else 'clause_pause_ms'
            count=round(rate*timing[key]/1000)
            report.append(dict(after=clause,start_sample=cursor,samples=count,pause_ms=timing[key]))
            parts.append(np.zeros(count));cursor+=count
    if not parts:raise ValueError('Empty prompt')
    return np.concatenate(parts),rate,dict(pauses=report)


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


def create_engine(config,cloud_key=None):
    if config['tts'].get('engine')=='dashscope':
        from vivis.speech.cloud_tts import CloudTTS
        return CloudTTS(config['tts'],secret=cloud_key)
    if config['tts'].get('engine')=='qwen3':
        from vivis.speech.qwen import QwenEngine
        return QwenEngine(config)
    import sherpa_onnx
    t=config['tts'];folder=ROOT/t['directory']
    common=dict(model=str(folder/t['model']),tokens=str(folder/t['tokens']),lexicon=str(folder/t['lexicon']))
    if t.get('engine','vits')=='kokoro':
        engine_model=dict(kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(**common,voices=str(folder/t['voices'])))
    elif t.get('engine','vits')=='vits':
        engine_model=dict(vits=sherpa_onnx.OfflineTtsVitsModelConfig(**common,dict_dir=str(folder/t['dict'])))
    else:raise ValueError('Unsupported local TTS engine')
    model=sherpa_onnx.OfflineTtsModelConfig(**engine_model,num_threads=t['threads'],provider='cpu')
    rules=','.join(str(folder/n) for n in t.get('rule_fsts',('date.fst','number.fst','phone.fst')))
    engine=sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=model,rule_fsts=rules))
    return engine


def prepare(config,include_diagnostics=False,cloud_key=None):
    if not include_diagnostics and prompt_cache_ready(config):
        print('Verified prepared prompt cache; no synthesis needed.',flush=True);return
    t=config['tts'];folder=ROOT/t['directory'] if 'directory' in t else None
    destination=ROOT/config['prompts'];destination.mkdir(parents=True,exist_ok=True)
    if t.get('engine')=='qwen3' and Path(sys.prefix).resolve()!=(ROOT/t['python']).parent.parent.resolve():
        job=destination/'prepare_config.json';job.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
        command=[str(ROOT/t['python']),'-m','vivis.speech.prompts','--prepare','--config',str(job)]
        if include_diagnostics:command+=['--diagnostics']
        subprocess.run(command,
            cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'PYTHONNOUSERSITE':'1','HF_HUB_OFFLINE':'1'},check=True)
        return
    texts=expected_texts(config)
    selected=texts if t.get('engine') not in ('qwen3','dashscope') or include_diagnostics else {name:texts[name] for name in RUNTIME_PROMPTS}
    def file_hash(path):
        h=hashlib.sha256()
        with path.open('rb') as src:
            for chunk in iter(lambda:src.read(1024*1024),b''):h.update(chunk)
        return h.hexdigest()
    metadata=dict(processing_version=PROCESSING_VERSION,tts=t,texts=selected,timing=config.get('prompt_timing'),asset_sha256={k:file_hash(folder/t[k])
        for k in ('model','tokens','voices','lexicon') if k in t and folder is not None})
    if 'source_manifest' in t:metadata['asset_sha256']['source_manifest']=file_hash(ROOT/t['source_manifest'])
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
    engine=create_engine(config,cloud_key=cloud_key) if t.get('engine')=='dashscope' else create_engine(config)
    if hasattr(engine,'execution_info'):metadata['generation_execution']=engine.execution_info
    sid=t.get('speaker_id',0);speed=t.get('speed',1.)
    if type(sid) is not int or not 0<=sid<engine.num_speakers or not .5<=speed<=2:raise ValueError('Invalid prompt speaker or speed')
    reports={};timing_reports={};metadata['prompt_sha256']={}
    for name,text in selected.items():
        path=destination/(name+'.wav')
        expected=cached.get('prompt_sha256',{}).get(name)
        if (path.is_file() and all(cached.get(k)==v for k,v in metadata.items() if k not in ('texts','prompt_sha256','generation_execution'))
                and cached.get('texts',{}).get(name)==text and (not expected or file_hash(path)==expected)):
            reports[name]=cached.get('level_reports',{}).get(name,{})
            timing_reports[name]=cached.get('timing_reports',{}).get(name,{})
            metadata['prompt_sha256'][name]=file_hash(path)
            continue
        samples,sample_rate,timing_reports[name]=synthesize_prompt(engine,text,sid,speed,config.get('prompt_timing'))
        if reference_channels>1:samples=np.repeat(samples[:,None],reference_channels,axis=1)
        temporary=path.with_suffix('.tmp.wav')
        with wave.open(str(temporary),'wb') as wav:
            wav.setnchannels(reference_channels);wav.setsampwidth(2);wav.setframerate(sample_rate)
            wav.writeframes(np.clip(samples*32767,-32768,32767).astype('<i2').tobytes())
        if level:
            samples,reports[name]=normalize_prompt(temporary,reference_loudness,level)
            with wave.open(str(temporary),'wb') as wav:
                wav.setnchannels(reference_channels);wav.setsampwidth(2);wav.setframerate(sample_rate)
                wav.writeframes(np.clip(samples*32767,-32768,32767).astype('<i2').tobytes())
        temporary.replace(path)
        metadata['prompt_sha256'][name]=file_hash(path)
        print('Prepared prompt: '+name,flush=True)
    (destination/'texts.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2)+'\n')
    metadata['level_reports']=reports
    metadata['timing_reports']=timing_reports
    temporary=destination/'metadata.tmp';temporary.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n');temporary.replace(destination/'metadata.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/perception/speech.json')
    parser.add_argument('--diagnostics',action='store_true',help='Also generate optional ASR test phrases with Qwen')
    parser.add_argument('--check-cache',action='store_true',help='Verify prepared prompts only; no models/devices')
    args=parser.parse_args()
    if args.prepare:prepare(json.loads(args.config.read_text()),args.diagnostics)
    if args.check_cache:
        if not prompt_cache_ready(json.loads(args.config.read_text())):raise SystemExit('Prompt cache missing, changed or incompatible')
        print('Prompt cache verified; no TTS model or device opened.')
