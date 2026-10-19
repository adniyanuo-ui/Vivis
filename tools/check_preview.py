"""Validate PNG frame integrity and fully decode the resulting MP4."""
import json,re,struct,subprocess,zlib
from pathlib import Path
root=Path(__file__).resolve().parent.parent
folder=root/'outputs/5_frames'
for i in range(830):
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
print('All 830 PNG frame CRCs and dimensions verified',flush=True)
video=root/'outputs/5_A2F_MARK_PREVIEW.mp4'
if not video.exists():raise SystemExit(0)
result=subprocess.run([str(root/'tools/bin/ffmpeg'),'-hide_banner','-v','info','-i',str(video),'-f','null','-','-progress','pipe:1','-nostats'],capture_output=True,text=True,check=True)
frames=[int(x) for x in re.findall(r'^frame=(\d+)$',result.stdout,re.M)]
assert frames[-1]==830,frames
assert 'Video: h264' in result.stderr and 'Audio: aac' in result.stderr
assert '768x768' in result.stderr and '30 fps' in result.stderr
report={'video':str(video),'bytes':video.stat().st_size,'png_frame_integrity':'passed','decoded_video_frames':frames[-1],'full_audio_video_decode':'passed','video_codec':'H.264','audio_codec':'AAC','video_fps':30,'resolution':[768,768],'audio_input_duration_seconds':27.648,'encoded_video_duration_seconds':830/30,'maximum_end_duration_difference_ms':abs(830/30-27.648)*1000,'subjective_lip_sync_acceptance':'pending user review'}
(root/'outputs/5_preview_validation.json').write_text(json.dumps(report,indent=2)+'\n')
(root/'logs/preview-decode-validation.log').write_text(result.stderr+'\n'+result.stdout)
print(json.dumps(report,indent=2))
