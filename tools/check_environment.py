"""Read-only checks for the local A2F development environment."""
import ctypes,json,platform,shutil,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent.parent
checks={'python':platform.python_version(),'workspace':str(root)}
for name in ['nvcc','cmake','ninja','ffmpeg','git','git-lfs','blender']:
    checks[name]=shutil.which(name)
import numpy, scipy, onnx, tensorrt
checks.update(numpy=numpy.__version__,scipy=scipy.__version__,onnx=onnx.__version__,tensorrt=tensorrt.__version__)
cuda=ctypes.CDLL('libcuda.so.1')
assert cuda.cuInit(0)==0
count=ctypes.c_int()
assert cuda.cuDeviceGetCount(ctypes.byref(count))==0
checks['cuda_devices']=count.value
for device in range(count.value):
    name=ctypes.create_string_buffer(128)
    assert cuda.cuDeviceGetName(name,128,device)==0
    checks[f'gpu_{device}']=name.value.decode()
print(json.dumps(checks,ensure_ascii=False,indent=2))
assert all(checks[x] for x in ['nvcc','cmake','ninja','ffmpeg','git-lfs','blender'])
