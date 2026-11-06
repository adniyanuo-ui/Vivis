"""Optional local Chinese TTS for cached menu prompts, never device playback."""
import argparse
import hashlib
import json
from pathlib import Path
import wave
import numpy as np
from vivis.paths import ROOT

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


def prepare(config):
    import sherpa_onnx
    t=config['tts'];folder=ROOT/t['directory'];destination=ROOT/config['prompts'];destination.mkdir(parents=True,exist_ok=True)
    metadata=dict(model_sha256=hashlib.sha256((folder/t['model']).read_bytes()).hexdigest(),texts=TEXTS)
    try:cached=json.loads((destination/'metadata.json').read_text())
    except (OSError,ValueError):cached={}
    vits=sherpa_onnx.OfflineTtsVitsModelConfig(model=str(folder/t['model']),tokens=str(folder/t['tokens']),
        lexicon=str(folder/t['lexicon']),dict_dir=str(folder/t['dict']))
    model=sherpa_onnx.OfflineTtsModelConfig(vits=vits,num_threads=t['threads'],provider='cpu')
    rules=','.join(str(folder/n) for n in ('date.fst','number.fst','phone.fst'))
    engine=sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=model,rule_fsts=rules))
    for name,text in TEXTS.items():
        path=destination/(name+'.wav')
        if (path.is_file() and cached.get('model_sha256')==metadata['model_sha256']
                and cached.get('texts',{}).get(name)==text):continue
        audio=engine.generate(text,sid=0,speed=1.0)
        with wave.open(str(path),'wb') as wav:
            wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(audio.sample_rate)
            wav.writeframes(np.clip(np.asarray(audio.samples)*32767,-32768,32767).astype('<i2').tobytes())
        print('Prepared prompt: '+name,flush=True)
    (destination/'texts.json').write_text(json.dumps(TEXTS,ensure_ascii=False,indent=2)+'\n')
    (destination/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare(json.loads((ROOT/'configs/perception/speech.json').read_text()))
