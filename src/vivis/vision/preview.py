"""Small local preview for camera-only acceptance. No stored face images by default."""
import json
import ipaddress
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from vivis.vision.interaction_log import interaction_snapshot

PAGE = '''<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>机器人 · 人脸跟随验证</title><style>
body{font:17px system-ui;background:#141b24;color:#e4edf7;max-width:1440px;margin:20px auto;padding:15px}
.monitor{display:grid;grid-template-columns:minmax(0,1fr) minmax(320px,1fr);gap:24px}
@media(max-width:850px){.monitor{grid-template-columns:1fr}}
#interaction-lines{height:520px;overflow:auto;background:#202c3b;padding:14px;border-radius:10px}
.log-line{white-space:pre-wrap;overflow-wrap:anywhere;margin:0 0 16px}.log-label{color:#8bc9fa;font-size:14px}
img{max-width:100%;border-radius:12px;background:#253142}small{color:#aac0d8}
.eye{display:inline-block;position:relative;width:100px;height:60px;border-radius:50%;background:white;margin:14px}
.pupil{position:absolute;width:24px;height:24px;border-radius:50%;background:#145177;left:38px;top:18px}
pre{white-space:pre-wrap;background:#202c3b;padding:16px;border-radius:10px}</style>
<h1>机器人 · 人脸跟随验证</h1><p>绿色框：面积最大的人脸；绿色十字：注视点；蓝色十字：画面中心。</p>
<div class="monitor"><section>
<img id="camera" alt="等待 USB 摄像头画面"><p><small id="camera-status">正在连接画面…</small></p>
<div><span class="eye"><i class="pupil"></i></span><span class="eye"><i class="pupil"></i></span></div>
<small>下方眼球是图像方向模拟，不代表舵机实测位置。此视觉服务不驱动机器人。</small>
<p id="robot-control">视觉服务只检测人脸，真机控制状态等待更新。</p>
<p id="identity-status"></p></section><section>
<h2>交互日志</h2><p id="interaction-status">正在读取…</p><p id="business-identity"></p>
<label><input id="log-follow" type="checkbox" checked> 自动滚动到最新</label>
<p><small>最近100条，本机只读。回复文字不代表音频已完整播出；可能包含私人对话，请勿公开分享页面。</small></p>
<div id="interaction-lines" role="log"></div></section></div>
<details><summary>检测详细数据</summary><pre id="state">等待检测…</pre></details><script>
let previousLog='';
async function refreshInteraction(){
const controller=new AbortController(),deadline=setTimeout(()=>controller.abort(),2000);
try{
const response=await fetch('/interaction-log',{cache:'no-store',signal:controller.signal});
if(!response.ok)throw new Error('Interaction log unavailable');
const data=await response.json();document.querySelector('#interaction-status').textContent=data.status+(data.self_check?' · '+data.self_check:'');
document.querySelector('#business-identity').textContent=data.identity_status||'';
const signature=JSON.stringify([data.session,data.entries]),box=document.querySelector('#interaction-lines');
if(signature!==previousLog){
const oldTop=box.scrollTop,fragment=document.createDocumentFragment();
for(const entry of data.entries){const row=document.createElement('div');row.className='log-line';
const label=document.createElement('div');label.className='log-label';label.textContent=entry.time+' · '+entry.role;
const content=document.createElement('div');content.textContent=entry.text;row.append(label,content);fragment.append(row);}
box.replaceChildren(fragment);box.scrollTop=document.querySelector('#log-follow').checked?box.scrollHeight:oldTop;previousLog=signature;}
}catch(e){document.querySelector('#interaction-status').textContent='交互日志连接中，正在重试…';}
finally{clearTimeout(deadline);setTimeout(refreshInteraction,1000);}}
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
const identity=s.familiar_identity,observation=identity&&identity.observation,age=observation&&observation.age_estimate;
const decision=observation&&observation.decision_reason,reasons={quality_low:'亮度或清晰度不足',no_feature:'未获得特征',below_threshold:'比对分数不足',ambiguous_candidates:'候选差距不足',confirming:'等待连续确认',confirmed:'已确认'};
document.querySelector('#identity-status').textContent=identity?
 '相机原始识别（未经过交互防抖）：'+(identity.backend||'未知')+'；状态：'+({recognized:'已确认',reviewing:'复核中',unknown:'未确认'}[identity.state]||identity.state)+
 (identity.person_id?'；ID：'+identity.person_id:'')+(decision?'；原因：'+(reasons[decision]||decision):'')+
 (observation&&observation.confirmation_hits!==undefined?'；连续确认：'+observation.confirmation_hits+'/'+observation.confirmation_required:'')+
 (age?'；外观年龄估计约'+age.years+'岁（不作身份依据）':''):'';
let control=s.robot_control,expression=control&&control.program==='expression-follow',held=expression&&control.eye_neck_held;
document.querySelector('#robot-control').textContent=control?(control.active?(expression?(held?'表情跟随：面部＋眼睑；眼球、颈部保持原位。剩余约 ':'表情＋眼球＋颈部联合跟随。剩余约 '):'真机跟踪正在运行，剩余约 ')+Math.ceil(control.remaining_seconds||0)+' 秒':'真机控制未运行；画面仍在检测人脸'):'视觉服务不驱动机器人';
let xy=held?[0,0]:(s.simulated_xy||[0,0]);document.querySelectorAll('.pupil').forEach(p=>p.style.transform=`translate(${xy[0]*29}px,${xy[1]*14}px)`);
}catch(e){document.querySelector('#state').textContent='视觉服务未连接，正在自动重试…';}
finally{clearTimeout(deadline);setTimeout(refreshState,250);}}
refreshCamera();refreshState();refreshInteraction();</script></html>'''


class Preview:
    def __init__(self, host, port, page=PAGE):
        self.lock = threading.Lock()
        self.jpeg = None
        self.state = {'status': 'starting'}
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                try:
                    if self.path == '/interaction-log':
                        if not ipaddress.ip_address(self.client_address[0]).is_loopback:
                            self.send_error(403)
                            return
                        data = json.dumps(interaction_snapshot(), ensure_ascii=False).encode()
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json; charset=utf-8')
                        self.send_header('Content-Length', str(len(data)))
                        self.send_header('Cache-Control', 'no-store')
                        self.end_headers()
                        self.wfile.write(data)
                        return
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
                            data, kind = page.encode(), 'text/html; charset=utf-8'
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
