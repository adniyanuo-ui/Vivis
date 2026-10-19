"""Validate PNG frame integrity and fully decode the resulting MP4."""
import json,re,struct,subprocess,zlib,sys,wave,math
from pathlib import Path
root=Path(__file__).resolve().parent.parent
folder=Path(sys.argv[1]).resolve()/'frames'
video=Path(sys.argv[2]).resolve()
video_width=int(sys.argv[3]) if len(sys.argv)>3 else 768
context=json.loads((folder.parent/'metadata.json').read_text())
audio_path=Path(context.get('audio',str(root/'5.wav')))
if not audio_path.is_absolute():audio_path=root/audio_path
with wave.open(str(audio_path)) as wav:audio_duration=wav.getnframes()/wav.getframerate()
expected_frames=math.ceil(audio_duration*30)
for i in range(expected_frames):
    p=folder/f'{i:04d}.png'
    data=p.read_bytes()
    assert data[:8]==b'\x89PNG\r\n\x1a\n',p
    cursor=8;finished=False
    while cursor<len(data):
        length=struct.unpack('>I',data[cursor:cursor+4])[0]
        kind=data[cursor+4:cursor+8]
        content=data[cursor+8:cursor+8+length]
        crc=struct.unpack('>I',data[cursor+8+length:cursor+12+length])[0]
        assert zlib.crc32(kind+content)&0xffffffff==crc,p
        if kind==b'IHDR':assert struct.unpack('>II',content[:8])==(768,768),p
        cursor+=length+12
        if kind==b'IEND':finished=True;break
    assert finished and cursor==len(data),p
print(f'All {expected_frames} PNG frame CRCs and dimensions verified',flush=True)

if not video.exists():raise SystemExit(0)
result=subprocess.run([str(root/'tools/bin/ffmpeg'),'-hide_banner','-v','info','-i',str(video),'-f','null','-','-progress','pipe:1','-nostats'],capture_output=True,text=True,check=True)
frames=[int(x) for x in re.findall(r'^frame=(\d+)$',result.stdout,re.M)]
assert frames[-1]==expected_frames,frames
assert 'Video: h264' in result.stderr and 'Audio: aac' in result.stderr
assert f'{video_width}x768' in result.stderr and '30 fps' in result.stderr
report={'video':str(video),'bytes':video.stat().st_size,'png_frame_integrity':'passed','decoded_video_frames':frames[-1],'full_audio_video_decode':'passed','video_codec':'H.264','audio_codec':'AAC','video_fps':30,'resolution':[video_width,768],'audio_input_duration_seconds':audio_duration,'encoded_video_duration_seconds':expected_frames/30,'maximum_end_duration_difference_ms':abs(expected_frames/30-audio_duration)*1000,'subjective_lip_sync_acceptance':'pending user review'}
(root/'outputs'/(video.stem+'_validation.json')).write_text(json.dumps(report,indent=2)+'\n')
(root/'logs'/(video.stem+'_decode.log')).write_text(result.stderr+'\n'+result.stdout)
print(json.dumps(report,indent=2))
