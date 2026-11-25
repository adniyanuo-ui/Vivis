"""Loopback-only administrator UI, independent of head/voice service lifecycle."""
import argparse
import json
import mimetypes
from pathlib import Path
import re
import secrets
import sqlite3
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

from vivis.paths import ROOT
from vivis.interaction.review_data import ReviewData
from vivis.speech.speaker_gallery import RevisionConflict as SpeakerConflict
from vivis.vision.photo_management import RevisionConflict as PhotoConflict


class ReviewServer(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,root=ROOT,port=8766):
        self.data=ReviewData(root);self.token=secrets.token_urlsafe(32);self.media={}
        from vivis.interaction.review_jobs import VoiceImportJobs
        self.imports=VoiceImportJobs(root)
        self.page=Path(__file__).with_name('review_page.html').read_bytes()
        super().__init__(('127.0.0.1',port),ReviewHandler)

    def server_close(self):
        self.imports.close();super().server_close()


class ReviewHandler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass  # Do not log personal paths/recordings.

    def trusted_host(self):
        port=self.server.server_port
        return self.headers.get('Host') in (f'127.0.0.1:{port}',f'localhost:{port}')

    def send_body(self,code,body,mime='application/json; charset=utf-8',extra=None):
        if isinstance(body,dict):body=json.dumps(body,ensure_ascii=False).encode()
        self.send_response(code);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Referrer-Policy','no-referrer')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self'; media-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        for k,v in (extra or {}).items():self.send_header(k,v)
        self.end_headers();self.wfile.write(body)

    def do_GET(self):
        if not self.trusted_host():return self.send_body(403,dict(error='仅供本机使用'))
        try:
            if self.path=='/':return self.send_body(200,self.server.page,'text/html; charset=utf-8')
            if self.path=='/api/state':
                state,media=self.server.data.snapshot();self.server.media=media;state['token']=self.server.token
                state['voice_imports']=self.server.imports.snapshot()
                return self.send_body(200,state)
            if re.fullmatch(r'/api/import/[a-f0-9]{32}',self.path):
                return self.send_body(200,self.server.imports.get(self.path.rsplit('/',1)[-1]))
            if re.fullmatch(r'/media/[a-f0-9]{64}',self.path):
                key=self.path.removeprefix('/media/');path=self.server.media.get(key)
                if path is None:
                    _,media=self.server.data.snapshot();self.server.media=media;path=media.get(key)
                if path is None:return self.send_body(404,dict(error='录音或照片不存在'))
                path=self.server.data.private_path(path)
                return self.send_media(path)
            return self.send_body(404,dict(error='页面不存在'))
        except (OSError,ValueError,KeyError,TypeError,sqlite3.Error) as exc:
            return self.send_body(400,dict(error='无法读取样本清单：'+type(exc).__name__))

    def send_media(self,path):
        # Media IDs resolve only to registered sample/round files. Never a path
        # supplied by the browser, and no unrestricted private-directory route.
        with path.open('rb') as f:
            f.seek(0,2);size=f.tell();start=0;end=size-1;code=200
            value=self.headers.get('Range')
            if value:
                match=re.fullmatch(r'bytes=(\d+)-(\d*)',value)
                if not match:return self.send_body(416,dict(error='不支持该读取范围'),extra={'Content-Range':f'bytes */{size}'})
                start=int(match[1]);end=min(int(match[2]),size-1) if match[2] else size-1;code=206
                if not 0<=start<=end<size:return self.send_body(416,dict(error='范围无效'),extra={'Content-Range':f'bytes */{size}'})
            self.send_response(code);self.send_header('Content-Type',mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
            self.send_header('Content-Length',str(max(0,end-start+1)));self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff');self.send_header('Accept-Ranges','bytes')
            if code==206:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            self.end_headers();f.seek(start);remaining=end-start+1
            while remaining>0:
                chunk=f.read(min(65536,remaining))
                if not chunk:break
                self.wfile.write(chunk);remaining-=len(chunk)

    def do_POST(self):
        origin=self.headers.get('Origin');port=self.server.server_port
        if (not self.trusted_host() or origin not in (f'http://localhost:{port}',f'http://127.0.0.1:{port}')
                or not secrets.compare_digest(self.headers.get('X-Vivis-Review-Token',''),self.server.token)):
            return self.send_body(403,dict(error='请在本机审核页操作并刷新页面'))
        if self.path not in ('/api/action','/api/import'):return self.send_body(404,dict(error='操作不存在'))
        if self.headers.get_content_type()!='application/json':return self.send_body(415,dict(error='需要JSON请求'))
        request=None
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=8192:raise ValueError('请求大小无效')
            self.connection.settimeout(5)
            request=json.loads(self.rfile.read(length))
            if self.path=='/api/import':
                result=self.server.imports.start(self.server.data.voice_import_request(request))
                return self.send_body(202,result)
            return self.send_body(200,self.server.data.apply(request))
        except (SpeakerConflict,PhotoConflict) as exc:return self.send_body(409,dict(error=str(exc)))
        except (OSError,ValueError,KeyError,TypeError,sqlite3.Error) as exc:
            if self.path=='/api/import':self.server.imports.reject(request,str(exc))
            return self.send_body(400,dict(error=str(exc)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8766);a=p.parse_args()
    if not 1024<=a.port<=65535:p.error('端口应为1024..65535')
    server=ReviewServer(port=a.port)
    print(f'本机样本管理：http://127.0.0.1:{server.server_port}/；不使用相机、麦克风或串口。',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
