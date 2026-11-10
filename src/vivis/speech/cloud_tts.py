"""Explicit preparation-only Qwen-TTS adapter. Runtime needs only cached WAVs."""
import argparse
import copy
import getpass
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import urllib.error
import urllib.parse
import urllib.request
import warnings

import numpy as np
from scipy.io import wavfile

from vivis.paths import ROOT


def decode_wav(raw):
    """Normalize valid cloud PCM/float WAV layouts to mono float samples."""
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always',wavfile.WavFileWarning)
        try:rate,values=wavfile.read(io.BytesIO(raw))
        except (ValueError,OSError,wavfile.WavFileWarning) as exc:
            raise ValueError('Invalid cloud WAV: '+str(exc)) from None
    # Streaming WAVs may carry an unknown/huge RIFF length even when the
    # downloaded body is complete. Use bounded decoded samples in that case;
    # reject ordinary truncated files rather than caching incomplete speech.
    declared=int.from_bytes(raw[4:8],'little') if raw[:4]==b'RIFF' else 0
    if any('EOF' in str(n.message) for n in notices) and declared<=8*1024*1024:
        raise ValueError('Truncated cloud WAV')
    channels=1 if values.ndim==1 else values.shape[1] if values.ndim==2 else 0
    source=dict(sample_rate_hz=int(rate),channels=channels,dtype=str(values.dtype),frames=len(values))
    if notices:source['header_warnings']=[str(n.message) for n in notices]
    if not 8000<=rate<=96000 or channels not in (1,2) or not 0<len(values)<=rate*60:
        raise ValueError('Cloud WAV outside allowed format/duration: '+str(source))
    if values.dtype==np.uint8:samples=(values.astype(float)-128)/128
    elif values.dtype.kind=='i' and values.dtype.itemsize in (2,4,8):
        samples=values.astype(float)/(2**(values.dtype.itemsize*8-1))
    elif values.dtype.kind=='f':samples=values.astype(float)
    else:raise ValueError('Unsupported cloud WAV encoding: '+str(source))
    if channels==2:samples=samples.mean(axis=1)
    if not np.isfinite(samples).all() or not np.any(samples) or np.max(np.abs(samples))>1.0001:
        raise ValueError('Invalid/silent cloud audio samples: '+str(source))
    return samples,int(rate),source


class CloudTTS:
    num_speakers=1

    def __init__(self,settings,secret=None,cache_directory=None):
        self.settings=settings
        self.cache_directory=(ROOT/'outputs/speech/cloud_prompt_sources' if cache_directory is None
                              else Path(cache_directory) if cache_directory is not False else None)
        self.secret=secret if secret is not None else os.environ.get(settings['api_key_env'],'')
        if not self.secret or any(c.isspace() for c in self.secret):raise ValueError('请通过--ask-key或环境变量提供有效百炼API Key')
        parts=urllib.parse.urlsplit(settings['endpoint'])
        if (parts.scheme!='https' or parts.hostname not in ('dashscope.aliyuncs.com','dashscope-intl.aliyuncs.com')
                or parts.path!='/api/v1/services/aigc/multimodal-generation/generation'
                or parts.username or parts.password or parts.query or parts.fragment):
            raise ValueError('Unsupported cloud TTS endpoint')
        self.execution_info=dict(provider='DashScope HTTP',model=settings['model'],voice=settings['voice'],request_ids=[],audio_formats=[])

    def request(self,request,maximum):
        try:
            with urllib.request.urlopen(request,timeout=self.settings.get('timeout_s',60)) as response:
                data=response.read(maximum+1)
            if len(data)>maximum:raise ValueError('Cloud TTS response exceeds bounded size')
            return data
        except urllib.error.HTTPError as exc:
            try:body=json.loads(exc.read(8192));detail=str(body.get('code',''))+' '+str(body.get('message',''))
            except (ValueError,OSError):detail='request rejected'
            raise RuntimeError(f'Cloud TTS HTTP {exc.code}: '+detail.replace(self.secret,'[redacted]')) from None
        except (OSError,ValueError) as exc:
            raise RuntimeError('Cloud TTS request failed: '+str(exc).replace(self.secret,'[redacted]')) from None

    def generate(self,text,sid=0,speed=1.):
        if sid!=0 or speed!=1. or not isinstance(text,str) or not 1<=len(text)<=600:
            raise ValueError('Cloud cached prompts require speaker0/native speed1.0/text1..600')
        payload=dict(model=self.settings['model'],input=dict(text=text,voice=self.settings['voice'],
                     language_type=self.settings.get('language','Chinese')))
        identity=hashlib.sha256(json.dumps([self.settings['endpoint'],payload],sort_keys=True).encode()).hexdigest()
        cached=self.cache_directory/(identity+'.wav') if self.cache_directory is not None else None
        if cached is not None and cached.is_file():
            raw=cached.read_bytes();result=json.loads(cached.with_suffix('.json').read_text())
            if len(raw)>8*1024*1024 or hashlib.sha256(raw).hexdigest()!=result.get('sha256'):
                raise ValueError('Raw cloud prompt cache integrity mismatch: '+str(cached))
        else:
            request=urllib.request.Request(self.settings['endpoint'],data=json.dumps(payload).encode(),
                headers={'Authorization':'Bearer '+self.secret,'Content-Type':'application/json'})
            result=json.loads(self.request(request,65536))
            if result.get('code') or result.get('status_code',200)!=200:
                raise RuntimeError('Cloud TTS rejected request: '+str(result.get('code','unknown')).replace(self.secret,'[redacted]'))
            url=result['output']['audio']['url'];parts=urllib.parse.urlsplit(url)
            if (parts.scheme not in ('http','https') or not parts.hostname or not parts.hostname.endswith('.aliyuncs.com')
                    or parts.username or parts.password or parts.port not in (None,80,443)):
                raise ValueError('Unexpected cloud audio download host')
            # Download separately: the API authorization is never sent to object storage.
            url=urllib.parse.urlunsplit(('https',parts.netloc,parts.path,parts.query,''))
            raw=self.request(urllib.request.Request(url),8*1024*1024)
            if cached is not None:
                cached.parent.mkdir(parents=True,exist_ok=True)
                temporary=cached.with_suffix('.tmp');temporary.write_bytes(raw);temporary.replace(cached)
                cached.with_suffix('.json').write_text(json.dumps(dict(request_id=result.get('request_id'),
                    sha256=hashlib.sha256(raw).hexdigest(),request=payload),ensure_ascii=False,indent=2)+'\n')
        samples,rate,source=decode_wav(raw)
        self.execution_info['audio_formats'].append(source)
        self.execution_info['request_ids'].append(result.get('request_id'))
        return SimpleNamespace(samples=samples,sample_rate=rate)


def main(argv=None):
    from vivis.speech.prompts import prepare,prompt_cache_ready
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/perception/speech.json')
    parser.add_argument('--preset',type=Path,default=ROOT/'configs/perception/prompt_voice_ethan.json')
    parser.add_argument('--ask-key',action='store_true');parser.add_argument('--activate',action='store_true')
    args=parser.parse_args(argv)
    original=json.loads(args.config.read_text());candidate=copy.deepcopy(original)
    preset=json.loads(args.preset.read_text())
    if (not {'tts','prompts'}<=set(preset) or not set(preset)<={'tts','prompts','prompt_text_overrides'}
            or preset['tts'].get('engine')!='dashscope'):parser.error('Invalid cloud voice preset')
    candidate.update(preset)
    if not prompt_cache_ready(candidate):
        name=candidate['tts']['api_key_env'];secret=os.environ.get(name)
        if args.ask_key:
            if not sys.stdin.isatty():parser.error('--ask-key需要交互终端')
            secret=getpass.getpass('百炼API Key（仅生成提示音，不保存）: ')
        prepare(candidate,cloud_key=secret)
    if not prompt_cache_ready(candidate):raise RuntimeError('提示音校验失败，原配置保持不变')
    if args.activate:
        if json.loads(args.config.read_text())!=original:raise RuntimeError('准备过程中配置发生变化；缓存已保存，请重新执行以切换')
        temporary=args.config.with_suffix('.tmp');temporary.write_text(json.dumps(candidate,ensure_ascii=False,indent=2)+'\n')
        temporary.replace(args.config)
        print('晨煦提示音已校验并启用；请退出当前P1后重新启动。播放时完全使用本地WAV。')
    else:print('晨煦缓存已校验；加--activate可切换当前系统播报。')
    return 0


if __name__=='__main__':raise SystemExit(main())
