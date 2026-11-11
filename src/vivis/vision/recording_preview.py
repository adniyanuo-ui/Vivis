"""Local framing aid; guide marks are drawn on copies, never on captured frames."""
import threading
import time
from vivis.vision.preview import Preview

PAGE='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>人脸对照录像取景</title>
<style>body{background:#19212b;color:#eee;font:20px system-ui;margin:25px}img{max-width:100%}</style>
<h2>人脸对照录像：只取景，不控制机器头</h2>
<p>请让额头、双眼、鼻子和下巴完整入镜，尽量位于蓝框内。蓝框是取景参考，不是人脸检测结果。</p>
<p>站好后回到启动脚本的终端按 Enter，倒计时后开始录像。录像原片不带蓝框或文字。</p>
<img id="view"><p id="status">等待画面</p><script>
async function tick(){try{const r=await fetch('/frame.jpg?t='+Date.now());if(!r.ok)throw Error();
const old=document.querySelector('#view').src;const url=URL.createObjectURL(await r.blob());
document.querySelector('#view').src=url;if(old.startsWith('blob:'))URL.revokeObjectURL(old);
const s=await(await fetch('/state')).json();document.querySelector('#status').textContent=s.message;
}catch(e){document.querySelector('#status').textContent='画面未就绪，请检查相机';}setTimeout(tick,200);}tick();
</script></html>'''


class FramingPreview:
    def __init__(self,camera,port=8765):
        self.camera=camera;self.phase='framing';self.error=None
        self.stop=threading.Event();self.ready=threading.Event()
        self.preview=Preview('127.0.0.1',port,page=PAGE)
        self.thread=threading.Thread(target=self._run,daemon=True,name='face-ab-framing');self.thread.start()

    def _run(self):
        import cv2
        try:
            while not self.stop.is_set():
                sample=self.camera.get()
                if sample is not None and 0<=time.monotonic()-sample[1]<.5:
                    image=sample[2].copy();h,w=image.shape[:2]
                    cv2.rectangle(image,(int(w*.15),int(h*.1)),(int(w*.85),int(h*.9)),(255,180,30),2)
                    ok,jpeg=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,85])
                    if not ok:raise RuntimeError('Cannot encode framing preview')
                    message={'framing':'取景中：整张脸入镜后，在终端按Enter',
                             'countdown':'倒计时：请保持位置', 'recording':'正在录制：请保持本段要求的姿态'}[self.phase]
                    self.preview.update(dict(status=self.phase,message=message,actuators_opened=False),jpeg.tobytes())
                    self.ready.set()
                else:self.preview.update(dict(status='camera_stale',message='等待相机新画面',actuators_opened=False))
                self.stop.wait(.15)
        except Exception as exc:self.error=type(exc).__name__;self.ready.set()

    def close(self):
        self.stop.set();self.thread.join(timeout=1);self.preview.close()
