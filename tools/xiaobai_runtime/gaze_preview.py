"""Small local preview for camera-only acceptance. No stored face images by default."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = '''<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>机器人 · 人脸跟随验证</title><style>
body{font:17px system-ui;background:#141b24;color:#e4edf7;max-width:980px;margin:30px auto;padding:15px}
img{max-width:100%;border-radius:12px;background:#253142}small{color:#aac0d8}
.eye{display:inline-block;position:relative;width:100px;height:60px;border-radius:50%;background:white;margin:14px}
.pupil{position:absolute;width:24px;height:24px;border-radius:50%;background:#145177;left:38px;top:18px}
pre{white-space:pre-wrap;background:#202c3b;padding:16px;border-radius:10px}</style>
<h1>机器人 · 人脸跟随验证</h1><p>绿色框：面积最大的人脸；绿色十字：注视点；蓝色十字：画面中心。</p>
<img id="camera" alt="等待 USB 摄像头画面"><p><small id="camera-status">正在连接画面…</small></p>
<div><span class="eye"><i class="pupil"></i></span><span class="eye"><i class="pupil"></i></span></div>
<small>仅左右跟随。下方眼球是方向模拟，不代表舵机实测位置。此视觉服务不驱动机器人。</small>
<p id="robot-control">视觉服务只检测人脸，真机控制状态等待更新。</p>
<pre id="state">等待检测…</pre><script>
const camera=document.querySelector('#camera'),cameraStatus=document.querySelector('#camera-status');
let displayedImage=null;
// A restarted service closes MJPEG streams; browsers do not reliably reconnect them.
// Fetch one fresh JPEG at a time, with a deadline and retry after disconnects.
async function refreshCamera(){
const controller=new AbortController(),deadline=setTimeout(()=>controller.abort(),2000);
let pendingImage=null;
try{
const response=await fetch('/frame.jpg?t='+Date.now(),{cache:'no-store',signal:controller.signal});
if(!response.ok)throw new Error('Camera image unavailable');
pendingImage=URL.createObjectURL(await response.blob());
camera.src=pendingImage;await camera.decode();
const previousImage=displayedImage;displayedImage=pendingImage;pendingImage=null;
if(previousImage)URL.revokeObjectURL(previousImage);
camera.style.opacity='1';cameraStatus.textContent='画面实时更新';
}catch(e){camera.style.opacity='.45';cameraStatus.textContent='画面连接中，正在自动重试…';}
finally{clearTimeout(deadline);if(pendingImage)URL.revokeObjectURL(pendingImage);setTimeout(refreshCamera,150);}
}
async function refreshState(){const controller=new AbortController(),deadline=setTimeout(()=>controller.abort(),2000);
try{let r=await fetch('/state',{cache:'no-store',signal:controller.signal});if(!r.ok)throw new Error('State unavailable');let s=await r.json();
document.querySelector('#state').textContent=JSON.stringify(s,null,2);
let control=s.robot_control;document.querySelector('#robot-control').textContent=control?(control.active?'真机跟踪正在运行，剩余约 '+Math.ceil(control.remaining_seconds||0)+' 秒':'真机跟踪未运行；画面仍在检测人脸'):'视觉服务不驱动机器人';
let xy=s.simulated_xy||[0,0];document.querySelectorAll('.pupil').forEach(p=>p.style.transform=`translate(${xy[0]*29}px,${xy[1]*14}px)`);
}catch(e){document.querySelector('#state').textContent='视觉服务未连接，正在自动重试…';}
finally{clearTimeout(deadline);setTimeout(refreshState,250);}}
refreshCamera();refreshState();</script></html>'''


class Preview:
    def __init__(self, host, port):
        self.lock = threading.Lock()
        self.jpeg = None
        self.state = {'status': 'starting'}
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                try:
                    if self.path == '/stream.mjpg':
                        self.send_response(200)
                        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                        self.send_header('Cache-Control', 'no-store')
                        self.end_headers()
                        while not owner.closed.is_set():
                            with owner.lock:
                                jpeg = owner.jpeg
                            if jpeg:
                                self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                                                 + str(len(jpeg)).encode() + b'\r\n\r\n' + jpeg + b'\r\n')
                                self.wfile.flush()
                            owner.closed.wait(.15)
                        return
                    with owner.lock:
                        if self.path == '/state':
                            payload = dict(owner.state)
                            payload['service_age_s'] = time.monotonic() - payload.pop('_updated_at', time.monotonic())
                            data, kind = json.dumps(payload, ensure_ascii=False).encode(), 'application/json'
                        elif self.path.split('?', 1)[0] == '/frame.jpg' and owner.jpeg:
                            data, kind = owner.jpeg, 'image/jpeg'
                        elif self.path == '/':
                            data, kind = PAGE.encode(), 'text/html; charset=utf-8'
                        else:
                            self.send_error(404)
                            return
                    self.send_response(200)
                    self.send_header('Content-Type', kind)
                    self.send_header('Content-Length', str(len(data)))
                    self.send_header('Cache-Control', 'no-store')
                    self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    pass

        self.closed = threading.Event()
        self.server = ThreadingHTTPServer((host, port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def update(self, state, jpeg=None):
        with self.lock:
            self.state = {**state, '_updated_at': time.monotonic()}
            if jpeg is not None:
                self.jpeg = jpeg

    def close(self):
        self.closed.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)
