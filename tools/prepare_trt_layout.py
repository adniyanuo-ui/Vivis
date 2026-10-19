from pathlib import Path
root=Path(__file__).resolve().parent.parent
site=root/'.venv/lib/python3.10/site-packages'
dest=root/'tools/tensorrt/lib';dest.mkdir(parents=True,exist_ok=True)
(root/'tools/tensorrt/include').mkdir(parents=True,exist_ok=True)
(root/'tools/bin').mkdir(parents=True,exist_ok=True)
for src in (root/'tools/tensorrt-debs/usr/lib/x86_64-linux-gnu').glob('*.so*'):
    p=dest/src.name
    if not p.exists():p.symlink_to(src)
    if '.so.' in src.name:
        p=dest/(src.name.split('.so.')[0]+'.so')
        if not p.exists():p.symlink_to(src)
for src in (root/'tools/tensorrt-debs/usr/include/x86_64-linux-gnu').glob('*.h'):
    p=root/'tools/tensorrt/include'/src.name
    if p.exists():p.unlink()
    p.symlink_to(src)
for folder in ['include','lib']:
    dest=root/'tools/cuda-12.9'/folder
    for src in (site/'nvidia/cublas'/folder).glob('*'):
        p=dest/src.name
        if not p.exists():p.symlink_to(src)
        if '.so.' in src.name:
            p=dest/(src.name.split('.so.')[0]+'.so')
            if not p.exists():p.symlink_to(src)
import imageio_ffmpeg
p=root/'tools/bin/ffmpeg'
if not p.exists():p.symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
print('TensorRT/CUDA/FFmpeg local layout ready')
