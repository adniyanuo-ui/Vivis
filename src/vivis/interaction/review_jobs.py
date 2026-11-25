"""One optional local voice-import subprocess; never blocks review or interaction."""
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid

from vivis.runtime.best_effort import BestEffortJob
from vivis.speech.speaker_gallery import commit


class VoiceImportJobs:
    def __init__(self,root):
        self.root=Path(root);self.job=BestEffortJob('review-voice-import')
        self.lock=threading.Lock();self.rows={};self.process=None
        self.receipts=self.root/'private/review/import_receipts.json'
        try:
            rows=json.loads(self.receipts.read_text())
            if not isinstance(rows,list):raise ValueError('Invalid review receipts')
            for row in rows[-32:]:
                if not isinstance(row,dict) or not isinstance(row.get('id'),str):continue
                row=dict(row)
                if row.get('state')=='running':row.update(state='unconfirmed',ok=False,error='上次选样未取得完成回执，请刷新声纹库核对；重试相同原话不会重复入库。')
                self.rows[row['id']]=row
        except (OSError,ValueError):pass  # Optional receipt history never blocks review.

    def snapshot(self):
        with self.lock:return [dict(row) for row in reversed(list(self.rows.values()))]

    def persist(self):
        try:
            self.receipts.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
            commit(self.receipts,list(self.rows.values())[-32:])
        except OSError:pass  # A history-write failure cannot change a committed import.

    def trim(self):
        while len(self.rows)>32:
            key=next((k for k,r in self.rows.items() if r.get('state')!='running'),None)
            if key is None:break
            self.rows.pop(key)

    def reject(self,request,error):
        if not isinstance(request,dict) or not isinstance(request.get('person'),str):return
        ids=request.get('round_ids');ids=[x for x in ids[:32] if isinstance(x,str)] if isinstance(ids,list) else []
        key=uuid.uuid4().hex
        with self.lock:
            self.rows[key]=dict(id=key,state='failed',ok=False,person=request['person'],source_turn_ids=ids,
                created=time.time(),error=error)
            self.trim();self.persist()

    def start(self,request):
        key=uuid.uuid4().hex
        with self.lock:
            self.rows[key]=dict(id=key,state='running',person=request.get('person'),source_turn_ids=request.get('round_ids',[]),
                created=time.time(),mode=request.get('mode','merge'),message='正在检查录音并生成候选，尚未保存。')
        if not self.job.offer(self.run,key,dict(request)):
            with self.lock:self.rows.pop(key,None)
            raise ValueError('另一个声纹导入正在处理，请稍后刷新核对')
        with self.lock:self.trim();self.persist()
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
            self.rows[key]=dict(self.rows[key],state='partial' if result.get('partial') else 'complete' if result['ok'] else 'failed',**result)
            self.trim()
            self.persist()

    def get(self,key):
        with self.lock:
            if key not in self.rows:raise ValueError('导入结果已失效，请刷新声纹库核对；相同原话不会重复入库')
            return dict(self.rows[key])

    def close(self):
        self.job.close()
        with self.lock:
            if self.process and self.process.poll() is None:self.process.terminate()
