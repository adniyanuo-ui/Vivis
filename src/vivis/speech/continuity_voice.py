"""Optional conversation continuity integration, isolated from motion/business."""
import asyncio
import json
import queue
import time

from vivis.paths import ROOT
from vivis.interaction.continuity import configuration


TOOL=dict(type='function',name='organize_archives',description='仅当当前用户明确要求整理聊天档案时调用。不是留言、节目或普通查询。程序核验管理员后，在待机中整理长期记忆；不删除近期聊天，不提前宣称成功。',
    parameters=dict(type='object',properties={},required=[],additionalProperties=False))


class ContinuityVoice:
    def init_continuity(self):
        self.continuity=None;self.archive_task=None;self.archive_notice_pending=set();self.archive_context_cache={};self.archive_context_at=0.
        c=configuration(self.speech['managed_voice'].get('continuity_config'),ROOT)
        if c and c['enabled']:
            from vivis.speech.round_recording import RoundRecorder
            self.continuity=RoundRecorder(c,ROOT,self.memory.path if self.memory else None)
            self.player.audio_observer=self.continuity.played
            self.session['tools']=[*self.session['tools'],TOOL]
            from vivis.speech.speaker_observation import SpeakerObserver
            try:
                self.continuity.speaker=SpeakerObserver.from_config(c.get('speaker_config'),ROOT,
                    lambda e:self.emit(dict(epoch=self.epoch,**e)),self.log.write)
                if self.continuity.speaker:
                    self.continuity.speaker.labels={pid:p.get('preferred_address') or p.get('display_name') or 'other'
                        for pid,p in self.knowledge.people.items()}
            except (OSError,ValueError) as exc:
                self.log.write('speaker_observer_unavailable',error_type=type(exc).__name__)
                self.emit(dict(kind='voice_warning',epoch=self.epoch,message='声纹观察未启用：'+type(exc).__name__+'；聊天继续。'))

    def continuity_start(self,event,timeline):
        if not self.continuity:return
        # Capture immutable raw evidence at the original turn; held logical
        # identity is sufficient for continuity, but never for voice enrollment.
        person=self.turn_person;evidence=self.continuity_evidence(person)
        self.continuity.start(event['item_id'],self.memory_session,person,timeline.at(event.get('audio_start_ms')),evidence)

    def continuity_evidence(self,person):
        identity=self.knowledge.current_identity();raw=(identity.get('person') or {}).get('person_id')
        reliable=(identity.get('state')=='recognized' and raw==person and self.recognized_person()==person
            and self.presence.person_count()<=1)
        evidence=dict(person_id=raw,confirmed=bool(reliable),single_face=bool(reliable))
        if getattr(self,'identity_fusion',None):
            evidence.update(voice_context=self.sync_voice_binding(),source_generation=self.input_generation,
                identity_evidence=self.fused_face_evidence())
        return evidence

    def collection_visual_evidence(self,now):
        """Raw selected-frame evidence for sampling only; no business identity update."""
        unknown=dict(stamp=now,person_id=None,confirmed=False,blocked=False,owner=None)
        if getattr(self,'sound_attention',None) is not None:return dict(unknown,blocked=True)
        config=getattr(self.knowledge,'recognition',None)
        if not config:return unknown
        try:
            face=json.loads((self.knowledge.root/config['state_file']).read_text())
            presence=json.loads(self.presence.path.read_text())
            if (not isinstance(face,dict) or face.get('schema_version')!='vivis-familiar-person-1.0'
                    or not isinstance(presence,dict) or presence.get('schema_version')!='vivis-face-presence-1.0'):return unknown
            stamp=face.get('observed_monotonic');visible=presence.get('observed_monotonic')
            if type(visible) not in (int,float) or not 0<=now-visible<=self.presence.c['max_age_s']:return unknown
            blocked=presence.get('faces',0)>1 or self.presence.person_count(now)>1
            identity_evidence=face.get('identity_evidence')
            if presence.get('faces')!=1:identity_evidence=None
            pid=face.get('person_id')
            if (face.get('state')=='recognized' and pid in self.knowledge.people and not blocked
                    and presence.get('faces')==1 and type(stamp) in (int,float)
                    and 0<=now-stamp<=config['max_result_age_s']):
                return dict(stamp=stamp,person_id=pid,confirmed=True,blocked=False,owner=face.get('owner'),identity_evidence=identity_evidence)
            return dict(unknown,blocked=blocked,owner=face.get('owner'),identity_evidence=identity_evidence)
        except (OSError,ValueError,TypeError,AttributeError):return unknown

    async def organize_archives(self,args,event):
        if not self.continuity:raise ValueError('聊天档案未启用')
        if args!={}:raise ValueError('整理档案不接受额外参数')
        self.fresh_task_source()
        if (self.context['mode']!='interaction' or not self.context['ready'] or self.in_speech
                or not self.pending_user_text or self.turn_person!=self.identified_person()
                or self.response_inputs.get(event.get('response_id'))!=self.input_generation):raise ValueError('请管理员在交互态明确提出整理档案')
        source=(self.input_generation,self.context['revision'],self.message_connection,self.memory_session,self.turn_item_id,self.turn_person)
        jid=await asyncio.to_thread(self.continuity.store.request_archive,self.identified_person())
        if source!=(self.input_generation,self.context['revision'],self.message_connection,self.memory_session,self.turn_item_id,self.turn_person):
            return dict(accepted=True,status='queued_without_state_change',job_id=jid,
                instruction='档案请求已入队，但新话轮已开始；暂不切换当前交互，待机时再整理，不重放旧回应。')
        rid=event.get('response_id')
        self.message_receipts.responses.pop(rid,None)
        if rid and rid not in self.blocked:self.blocked.append(rid)
        self.continue_response=False;self.tool_repair=None
        if rid and self.cloud_state:self.cloud_state.interrupt(rid,source='archive_ack_owned_by_controller')
        self.emit(dict(kind='archive_notice',epoch=self.epoch,revision=self.context['revision'],
            text='收到，我先休息一下，整理聊天档案。',job_id=jid,enter_standby=True))
        return dict(accepted=True,status='queued',job_id=jid,instruction='已排入待机整理，不是整理成功；完成或失败由程序播报。')

    async def archive_run(self,jid):
        store=self.continuity.store
        try:
            from vivis.speech.memory_compaction import extract_facts
            while self.context['mode']=='standby' and self.context['ready'] and not self.in_speech:
                batch=await asyncio.to_thread(store.next_batch,jid)
                if not batch:return
                result=await extract_facts(self.url,self.secret,self.session,batch,store.c['archive_timeout_s'])
                if self.context['mode']!='standby' or not self.context['ready'] or self.in_speech:
                    with store.db() as db:db.execute("UPDATE jobs SET state='paused' WHERE id=?",(jid,))
                    return
                await asyncio.to_thread(store.commit_batch,jid,batch,result)
                self.archive_context_at=0.
                self.log.write('archive_batch_committed',job_id=jid,person_id=batch['person_id'],rounds=len(batch['messages']))
        except asyncio.CancelledError:
            with store.db() as db:db.execute("UPDATE jobs SET state='paused' WHERE id=? AND state='running'",(jid,))
            raise
        except Exception as exc:
            error=type(exc).__name__+': '+str(exc)[:150]
            with store.db() as db:db.execute("UPDATE jobs SET state='failed',error=? WHERE id=?",(error,jid))
            self.log.write('archive_failed',job_id=jid,error=error)

    async def continuity_tick(self,state):
        try:await self._continuity_tick(state)
        except asyncio.CancelledError:raise
        except Exception as exc:
            if not getattr(self,'continuity_fault_reported',False):
                self.continuity_fault_reported=True
                self.log.write('continuity_unavailable',error_type=type(exc).__name__)
                self.emit(dict(kind='voice_warning',epoch=self.epoch,message='聊天档案暂不可用：'+type(exc).__name__+'；语音和留言继续使用。'))

    async def _continuity_tick(self,state):
        if not self.continuity:return
        now=time.monotonic()
        speaker=self.continuity.speaker
        if ((speaker and getattr(speaker,'collector',None) or getattr(self,'identity_fusion',None))
                and now>=getattr(self,'collection_visual_at',0.)):
            self.collection_visual_at=now+.2
            self.continuity.offer_sampling_identity(self.collection_visual_evidence,now)
        self.continuity.poll(state,self.memory_session)
        try:
            notice=self.continuity.notices.get_nowait()
            self.emit(dict(kind='voice_warning',epoch=self.epoch,message=notice))
        except queue.Empty:pass
        speaker=self.continuity.speaker
        if speaker:speaker.poll()
        now=time.monotonic()
        if now-self.archive_context_at<2:return
        self.archive_context_at=now
        store=self.continuity.store;person=self.identified_person()
        if person:
            self.archive_context_cache={person:await asyncio.to_thread(store.context,person)}
        else:self.archive_context_cache={}
        if self.context['mode']!='standby' or not self.context['ready'] or self.in_speech:return
        self.continuity.submit('settle')
        def jobs():
            with store.db() as db:
                return [dict(r) for r in db.execute("SELECT * FROM jobs WHERE notice=0 ORDER BY requested")]
        for job in await asyncio.to_thread(jobs):
            if job['state'] in ('queued','paused','running'):
                if self.archive_task is None or self.archive_task.done():self.archive_task=asyncio.create_task(self.archive_run(job['id']))
                break
            if job['id'] not in self.archive_notice_pending:
                text='档案更新完毕。' if job['state']=='complete' else '档案整理失败：'+(job['error'] or '原因未确认')+'。原记录保留。'
                self.archive_notice_pending.add(job['id'])
                self.emit(dict(kind='archive_notice',epoch=self.epoch,revision=self.context['revision'],text=text,job_id=job['id'],enter_standby=False))
                break
