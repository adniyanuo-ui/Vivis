"""One optional local voice-import subprocess; never blocks review or interaction."""
import json
import os
from pathlib import Path
import subprocess
import threading
import uuid

from vivis.runtime.best_effort import BestEffortJob


class VoiceImportJobs:
    def __init__(self,root):
        self.root=Path(root);self.job=BestEffortJob('review-voice-import')
        self.lock=threading.Lock();self.rows={};self.process=None

    def start(self,request):
        key=uuid.uuid4().hex
        with self.lock:
            self.rows[key]=dict(id=key,state='running',message='正在检查录音并生成候选，尚未保存。')
        if not self.job.offer(self.run,key,dict(request)):
            with self.lock:self.rows.pop(key,None)
            raise ValueError('另一个声纹导入正在处理，请稍后刷新核对')
        return dict(ok=True,job_id=key,message='已开始检查，尚未保存；结果以处理完成提示为准。')

    def run(self,key,request):
        process=None
        try:
            python=self.root/'outputs/speech/venv/bin/python'
            if not python.is_file():raise ValueError('本地语音运行环境不存在，未导入候选')
            env=dict(os.environ,PYTHONPATH=str(self.root/'src'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            process=subprocess.Popen([str(python),'-m','vivis.speech.speaker_import','--root',str(self.root)],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env,cwd=self.root)
            with self.lock:self.process=process
            output,_=process.communicate(json.dumps(request),timeout=120)
            result=json.loads(output)
            if not isinstance(result,dict) or type(result.get('ok')) is not bool:raise ValueError('未取得有效导入回执，请刷新样本库核对')
        except Exception as exc:
            if process and process.poll() is None:
                process.kill()
                try:process.communicate(timeout=2)
                except subprocess.TimeoutExpired:pass
            result=dict(ok=False,error=str(exc) if isinstance(exc,ValueError) else '导入未确认：'+type(exc).__name__+'；请刷新样本库核对后再试')
        with self.lock:
            self.process=None
            self.rows[key]=dict(id=key,state='complete' if result['ok'] else 'failed',**result)
            while len(self.rows)>32:self.rows.pop(next(iter(self.rows)))

    def get(self,key):
        with self.lock:
            if key not in self.rows:raise ValueError('导入结果已失效，请刷新声纹库核对；相同原话不会重复入库')
            return dict(self.rows[key])

    def close(self):
        self.job.close()
        with self.lock:
            if self.process and self.process.poll() is None:self.process.terminate()
