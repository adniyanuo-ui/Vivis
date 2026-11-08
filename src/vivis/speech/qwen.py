"""Optional local Qwen3-TTS adapter, loaded only in its separate environment."""
from pathlib import Path
from types import SimpleNamespace
import math
import numpy as np
from vivis.paths import ROOT


class QwenEngine:
    num_speakers=1  # Common prompt contract; the voice is selected by name.

    def __init__(self,config):
        import torch
        from qwen_tts import Qwen3TTSModel
        t=config['tts'];threads=t.get('threads',4)
        if type(threads) is not int or not 1<=threads<=12:raise ValueError('Invalid Qwen CPU threads')
        torch.set_num_threads(threads)
        device=t.get('device','cpu');dtype=t.get('dtype','float32')
        if device not in ('cpu','cuda:0','cuda:1') or dtype not in ('float32','float16','bfloat16'):
            raise ValueError('Invalid Qwen generation device/dtype')
        self.execution_info=dict(device=device,dtype=dtype,threads=threads)
        self.model=Qwen3TTSModel.from_pretrained(str(ROOT/t['directory']),
            device_map=device,dtype=getattr(torch,dtype),attn_implementation='sdpa',local_files_only=True)
        self.language=t.get('language','Chinese');self.max_tokens=t.get('max_new_tokens',1024)
        self.standard_chinese=t.get('force_standard_chinese',False)
        if type(self.standard_chinese) is not bool:raise ValueError('Invalid standard Chinese switch')
        if self.standard_chinese and self.language.lower()!='chinese':
            raise ValueError('Standard Chinese requires language=Chinese explicitly')
        if type(self.max_tokens) is not int or not 64<=self.max_tokens<=2048:
            raise ValueError('Invalid Qwen generation limit')
        self.set_voice(t['speaker_name'])

    def set_voice(self,voice):
        choices={v.lower():v for v in self.model.get_supported_speakers()}
        if not isinstance(voice,str) or voice.lower() not in choices:raise ValueError('Unknown Qwen voice: '+str(voice))
        self.voice=choices[voice.lower()]

    def generate(self,text,sid=0,speed=1.):
        if sid!=0 or not math.isfinite(speed) or speed!=1.:
            raise ValueError('Qwen0.6B uses a named voice and native speed1.0; no artificial rate change')
        talker=self.model.model.config.talker_config if self.standard_chinese else None
        original=talker.spk_is_dialect if talker is not None else None
        try:
            # The official wrapper otherwise overrides Chinese/Auto with the
            # preset's Beijing/Sichuan dialect token. Change only this in-memory
            # language selection; keep the downloaded weights/config unchanged.
            if talker is not None and self.language.lower()=='chinese':
                talker.spk_is_dialect={**original,self.voice:False}
            wavs,rate=self.model.generate_custom_voice(text=text,language=self.language,speaker=self.voice,
                max_new_tokens=self.max_tokens)
        finally:
            if talker is not None:talker.spk_is_dialect=original
        samples=np.asarray(wavs[0],dtype=np.float32)
        if samples.ndim!=1 or not len(samples) or not np.isfinite(samples).all():
            raise ValueError('Invalid Qwen synthesized audio')
        return SimpleNamespace(samples=samples,sample_rate=int(rate))
