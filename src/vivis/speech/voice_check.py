"""Local acoustic front-end check; no API key, network service or robot imports."""
import argparse
from contextlib import ExitStack
from datetime import datetime
import fcntl
import json
import math
from pathlib import Path
import queue
import time
import wave

import numpy as np

from vivis.paths import ROOT
from vivis.speech.realtime import Journal,check_audio_callbacks
from vivis.speech.realtime_audio import AudioIO,VoiceBuffer
from vivis.speech.voice_frontend import validate_frontend


PHRASE='小帅，请停止说话。今天阳光很好，我想和你聊一聊。'
STAGES=(('quiet',3),('near_only',10),('quiet',3),('far_only',30),('quiet',3),('double_talk',15),('quiet',3))
INSTRUCTIONS={
    'quiet':'安静，等待下一阶段。',
    'near_only':'现在请读两遍：'+PHRASE+'（音箱不播放）',
    'far_only':'现在仅音箱播放，请保持安静{duration}秒。',
    'double_talk':'现在音箱播放中，请立即重复读，不必等音箱说完：'+PHRASE,
}


def source_audio(seconds):
    """Use accepted male cached speech, with original level; no generated/cloud audio."""
    chunks=[]
    for name in ('identity','menu','clarify'):
        path=ROOT/'assets/speech/xiaoshuai_qwen3_mandarin'/(name+'.wav')
        with wave.open(str(path)) as wav:
            if (wav.getframerate(),wav.getnchannels(),wav.getsampwidth())!=(24000,1,2):
                raise ValueError('Local check needs24k mono PCM16: '+str(path))
            chunks.append(np.frombuffer(wav.readframes(wav.getnframes()),'<i2'))
        chunks.append(np.zeros(12000,dtype='<i2'))
    source=np.concatenate(chunks);count=int(seconds*24000)
    return np.tile(source,(count+len(source)-1)//len(source))[:count].astype('<i2').tobytes()


def run(speech,speaker,frontend,folder,stages=STAGES,gain=1.):
    log=Journal(folder/'events.jsonl');player=VoiceBuffer(gain=gain,max_seconds=max(duration for _,duration in stages)+1)
    audio=AudioIO(speech,speaker,player,True,frontend=frontend);started=None
    report=dict(kind='local_voice_frontend_check',actuators_opened=False,camera_opened=False,
                cloud_called=False,record_audio=True,frontend=frontend,output_gain=gain,phrase=PHRASE,stages=[],error=None)
    frame_index=0;epoch=0
    # Prepare all samples before device callbacks; never perform synthesis in a callback.
    try:
        sources={name:source_audio(duration) for name,duration in stages if name in ('far_only','double_talk')}
        with ExitStack() as stack:
            writers={}
            for name in ('microphone','reference','uplink'):
                wav=stack.enter_context(wave.open(str(folder/(name+'_16k.wav')),'wb'))
                wav.setparams((1,2,16000,0,'NONE','not compressed'));writers[name]=wav
            frames=stack.enter_context((folder/'frames.jsonl').open('w'))
            devices=audio.open();started=time.monotonic();report['devices']=devices
            log.write('audio_opened',devices=devices,frontend=frontend)
            ready_deadline=started+1.
            # First callbacks + processed frame must exist before the stage clock starts.
            while audio.last_capture is None or audio.last_output is None or audio.uplink.empty():
                if audio.fault:raise RuntimeError(audio.fault)
                check_audio_callbacks(audio,started,time.monotonic())
                if time.monotonic()>ready_deadline:raise RuntimeError('Local check did not receive its first processed frame')
                time.sleep(.005)
            for name,duration in stages:
                epoch+=1;rid='check-'+str(epoch);stage_start=time.monotonic()
                stage=dict(name=name,start_monotonic=stage_start,duration_s=duration)
                report['stages'].append(stage);log.write('stage',**stage)
                print(INSTRUCTIONS[name].format(duration=duration),flush=True)
                source=sources.get(name,b'');offset=0
                while time.monotonic()-stage_start<duration:
                    now=time.monotonic();check_audio_callbacks(audio,started,now)
                    if audio.fault:raise RuntimeError(audio.fault)
                    # Keep <1s lookahead. Resampling an entire30s clip while
                    # holding the player's callback lock can underrun the DAC.
                    if offset<len(source) and player.count<24000:
                        chunk=source[offset:offset+9600]  # 200ms at24k PCM16.
                        player.add(rid,chunk);offset+=len(chunk)
                        if offset==len(source):player.finish(rid)
                    while True:
                        try:stamp,clean,raw,far=audio.uplink.get_nowait()
                        except queue.Empty:break
                        if now-stamp>.5:raise RuntimeError('Local recording backlog exceeded500ms')
                        for key,data in [('microphone',raw),('reference',far),('uplink',clean)]:writers[key].writeframesraw(data)
                        frames.write(json.dumps(dict(sample=frame_index,adc_monotonic=stamp))+'\n');frame_index+=320
                    while not player.notices.empty():log.write(**player.notices.get_nowait())
                    while not audio.meters.empty():log.write(**audio.meters.get_nowait())
                    time.sleep(.005)
                stage['end_monotonic']=time.monotonic()
                # Stage transitions bound queued audio, with the same gentle cancellation as live chat.
                player.cancel(rid)
            print('本地采音完成。未连接云端、串口或相机；请保留录音供分析。',flush=True)
    except KeyboardInterrupt:report['operator_stop']=True
    except Exception as exc:report['error']=str(exc)
    finally:
        audio.close();report.update(audio.stats)
        report.update(recorded_samples=frame_index,recorded_s=frame_index/16000,
                      elapsed_s=time.monotonic()-started if started is not None else 0)
        (folder/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        log.close()
    if report['error']:print('本地检查失败：'+report['error'],flush=True)
    return 1 if report['error'] or report.get('operator_stop') else 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aec-backend',choices=('speex','aec3'),default='aec3')
    parser.add_argument('--stage',choices=('all','near-only','far-only','double-talk'),default='all')
    parser.add_argument('--seconds',type=int,default=30,help='单阶段时长10..60秒；all固定67秒')
    parser.add_argument('--output-gain',type=float,help='仅本次音箱数字幅度，不修改系统音量；0.5约为-6dB')
    parser.add_argument('--speaker');parser.add_argument('--mic');parser.add_argument('--output',type=Path)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args(argv)
    if not 10<=args.seconds<=60:parser.error('单阶段时长必须10..60秒')
    config=json.loads((ROOT/'configs/perception/realtime_voice.json').read_text())
    gain=config['output_gain'] if args.output_gain is None else args.output_gain
    if not math.isfinite(gain) or not .1<=gain<=2.:parser.error('output-gain必须是0.1..2.0的有限数')
    speech=json.loads((ROOT/config['speech']).read_text());frontend=dict(config.get('frontend',{}))
    frontend['backend']=args.aec_backend;validate_frontend(frontend)
    robot=json.loads((ROOT/config['robot']).read_text())
    hardware=json.loads((ROOT/config['robot']).with_name(robot['hardware']).read_text())
    speaker=args.speaker or hardware['runtime']['audio_device']
    if args.mic:speech['capture']['device']=args.mic
    stages=STAGES if args.stage=='all' else (('quiet',3),(args.stage.replace('-','_'),args.seconds),('quiet',3))
    if args.check:
        print(json.dumps(dict(valid=True,frontend=frontend,output_gain=gain,stages=stages,hardware_opened=False,cloud_called=False)))
        return 0
    folder=(args.output or ROOT/'outputs/voice_frontend'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    folder.mkdir(parents=True,exist_ok=False)
    print('本地音频前端检查（保存本机录音）：'+str(folder),flush=True)
    lock_path=ROOT/'outputs/runtime/p1.lock';lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:parser.error('P1或语音测试仍在运行，请先退出旧实例')
        return run(speech,speaker,frontend,folder,stages,gain=gain)


if __name__=='__main__':raise SystemExit(main())
