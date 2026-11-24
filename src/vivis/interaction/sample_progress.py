"""Report committed private sample manifests, with bounded optional background I/O."""
import json
from pathlib import Path

from vivis.runtime.best_effort import BestEffortJob


class SampleProgress:
    def __init__(self,root,emit):
        self.root=Path(root);self.emit=emit;self.previous={};self.next_poll=0.
        self.job=BestEffortJob('sample-progress');self.error=None

    def poll(self,now):
        if now<self.next_poll:return
        self.next_poll=now+2.
        self.job.offer(self.scan)

    def scan(self):
        try:
            photo_config=json.loads((self.root/'configs/perception/familiar_faces.json').read_text())
            voice_config=json.loads((self.root/'configs/perception/speaker_observation.json').read_text())
            self.voice_maximum=voice_config.get('maximum_samples',32)
            self.voice_minimum=voice_config.get('enrollment_samples',3)
            people=(self.root/photo_config['people_directory']).resolve()
            voices=(self.root/voice_config['directory']).resolve()
            private=(self.root/'private').resolve()
            if not people.is_relative_to(private) or not voices.is_relative_to(private):return
            profiles=list(people.glob('*/profile.json'))[:50]
            voice_file=voices/'gallery.json'
            stamp=voice_file.stat().st_mtime_ns if voice_file.exists() else None
            # Read the large feature document only when its revision/file changes.
            if stamp!=getattr(self,'voice_stamp',None):
                self.voice_rows=json.loads(voice_file.read_text()).get('people',{}) if stamp else {}
                self.voice_stamp=stamp
            for profile in profiles:
                pid=profile.parent.name;data=json.loads(profile.read_text())
                name=data.get('preferred_address') or data.get('display_name') or pid
                manifest=profile.with_name('photo_pool.json')
                photo_stamp=manifest.stat().st_mtime_ns if manifest.exists() else None
                key=('photo',pid)
                if photo_stamp!=self.previous.get(key,{}).get('stamp') or key not in self.previous:
                    pool=json.loads(manifest.read_text()) if manifest.exists() else dict(rows=[],active=[])
                    rows=[r for r in pool['rows'] if self.local_file(profile.parent,r['path'],people)]
                    ids={r['hash'] for r in rows};active=len(ids&set(pool['active']))
                    self.report(key,name,ids,active,photo_stamp)
                key=('voice',pid)
                if stamp!=self.previous.get(key,{}).get('stamp') or key not in self.previous:
                    rows=[r for r in getattr(self,'voice_rows',{}).get(pid,[])
                          if self.local_file(voices/pid,r['turn_id']+'.wav',voices)]
                    ids={r['turn_id'] for r in rows};active=sum(r.get('active',True) for r in rows)
                    self.report(key,name,ids,active,stamp)
            self.error=None
        except (OSError,ValueError,KeyError,TypeError) as exc:self.error=type(exc).__name__

    @staticmethod
    def local_file(folder,relative,allowed):
        path=(folder/relative).resolve()
        return path.is_relative_to(allowed) and path.is_file()

    def report(self,key,name,ids,active,stamp):
        previous=self.previous.get(key);added=len(ids-previous['ids']) if previous else 0
        kind=key[0];title='照片' if kind=='photo' else '声纹参考'
        changed=previous is None or previous['ids']!=ids or previous['active']!=active
        if changed:
            message=(title+'采集成功：' if added else title+'样本库：')+name+'，'+(f'新增{added}'+('张，' if kind=='photo' else '段，') if added else '')
            maximum=32 if kind=='photo' else self.voice_maximum
            message+=f'当前{len(ids)}/{maximum}；活动{active}'+('/16，候选'+str(len(ids)-active) if kind=='photo' else f'，至少{self.voice_minimum}段活动参考参与比对')
            if len(ids)>=maximum:message+='；已采满，停止自动新增'
            event=dict(kind='sample_progress',sample_type=kind,person_id=key[1],added=added,
                total=len(ids),active=active,committed=True,initial=previous is None,message=message)
            if self.emit(event) is False:return  # Retry next poll if the control queue is busy.
        self.previous[key]=dict(ids=ids,active=active,stamp=stamp)

    def close(self):self.job.close()
