"""Best-effort journal tap; no device owner, DB/disk work only on one worker.

PCM boundaries use the sent input stream's server offsets, not delayed event
arrival times. Output WAVs contain samples actually handed to the DAC callback.
"""
from collections import deque
import json
import queue
import fcntl
import os
import threading
import time
import uuid
import wave

import numpy as np

from vivis.interaction.continuity import RoundStore


class SentTimeline:
    def __init__(self):self.frames=deque(maxlen=9000);self.samples=0
    def sent(self,stamp,pcm):
        self.frames.append((self.samples,self.samples+len(pcm)//2,stamp));self.samples+=len(pcm)//2
    def at(self,milliseconds):
        if type(milliseconds) not in (int,float) or milliseconds<0:return None
        target=round(milliseconds*16)
        for start,end,stamp in reversed(self.frames):
            if start<=target<=end:return stamp+(target-start)/16000
        return None


class RoundRecorder:
    def __init__(self,config,root,legacy_path=None):
        self.store=RoundStore(config,root,legacy_path)
        self.owner_fd=os.open(self.store.directory/'journal.lock',os.O_CREAT|os.O_RDWR,0o600)
        try:fcntl.flock(self.owner_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self.owner_fd);raise RuntimeError('已有聊天档案记录进程，请先停止重复的交互进程') from None
        self.store.recover()
        self.queue=queue.Queue(maxsize=2400);self.notices=queue.SimpleQueue()
        self.closed=False;self.dropped=False;self.drop_generation=0;self.turns={};self.responses={};self.live={}
        self.ring=deque(maxlen=1500);self.active={};self.output={};self.last_poll=0.
        self.contexts={};self.speaker=None;self.previous_round=None;self.invalid_inputs={};self.finished_inputs={}
        self.admissions={};self.finalized={};self.source_keys={};self.blocked_people=set()
        self.sampling_points=deque(maxlen=400);self.sampling_epoch=0;self.sampling_person=None;self.sampling_owner=None
        self.visual_evidence=None;self.audio_epoch=0
        from vivis.runtime.best_effort import BestEffortJob
        self.sampling_job=BestEffortJob('voice-sampling-visual')
        self.thread=threading.Thread(target=self.work,name='conversation-journal',daemon=True);self.thread.start()

    def submit(self,kind,*args):
        if self.closed:return False
        try:self.queue.put_nowait((kind,args));return True
        except queue.Full:
            if not self.dropped:self.notices.put('记忆录音处理繁忙，本段记录可能不完整；不影响当前聊天和留言。')
            self.dropped=True;self.drop_generation+=1;return False

    def capture(self,stamp,pcm,far):self.submit('capture',stamp,pcm,far)
    def played(self,rid,values,stamp,position):self.submit('played',rid,values.copy(),stamp,position)

    def start(self,item,session,person,stamp,evidence=None):
        self.submit('start',item,session,person,stamp,evidence)
    def final(self,item,text,accepted=True):self.submit('final',item,text,accepted)
    def end(self,item,stamp,valid=True,evidence=None):self.submit('end',item,stamp,valid,evidence)
    def sampling_identity(self,evidence):self.submit('sampling_identity',dict(evidence))

    def offer_sampling_identity(self,reader,now):
        def read():
            try:self.sampling_identity(reader(now))
            except Exception:pass  # Optional evidence expires without affecting voice/control.
        return self.sampling_job.offer(read)

    def invalidate_sampling(self):
        self.sampling_epoch+=1
        if self.speaker and hasattr(self.speaker,'set_collection_epoch'):self.speaker.set_collection_epoch(self.sampling_epoch)

    def on_invalidate_sampling(self):self.invalidate_sampling()

    def invalidate_identity_audio(self):
        """Only capture/connection invalidation revokes voice evidence."""
        self.audio_epoch+=1;self.invalidate_sampling()
        if self.speaker and hasattr(self.speaker,'set_audio_epoch'):self.speaker.set_audio_epoch(self.audio_epoch)

    def on_invalidate_identity_audio(self):self.invalidate_identity_audio()

    def on_sampling_identity(self,evidence):
        stamp=evidence.get('stamp');pid=evidence.get('person_id');owner=evidence.get('owner')
        if type(stamp) not in (int,float) or not np.isfinite(stamp):return
        self.visual_evidence=dict(evidence)
        changed=owner and self.sampling_owner and owner!=self.sampling_owner
        changed=changed or evidence.get('blocked') or bool(pid and self.sampling_person and pid!=self.sampling_person)
        if changed:self.sampling_epoch+=1
        if owner:self.sampling_owner=owner
        if pid:self.sampling_person=pid
        point=dict(evidence,epoch=self.sampling_epoch)
        if not self.sampling_points or point!=self.sampling_points[-1]:self.sampling_points.append(point)
        if self.speaker and hasattr(self.speaker,'set_collection_epoch'):self.speaker.set_collection_epoch(self.sampling_epoch)

    def offer_speaker(self,clip):
        if not self.speaker:return
        from vivis.speech.speaker_sampling import visual_ranges
        start=clip.get('speaker_start',clip['start']);end=start+len(b''.join(clip['near']))/32000
        config=getattr(self.speaker,'c',{})
        seconds=((config.get('short_collection') or {}).get('visual_witness_s',.5) if isinstance(config,dict) else .5)
        ranges,conflict=visual_ranges(list(self.sampling_points),start,end,seconds)
        # Existing fully verified turns remain eligible as a whole; business
        # attribution is never rewritten when only interior frames identify a face.
        evidence=clip.get('evidence') or {};pid=evidence.get('person_id')
        if (not conflict and clip.get('sampling_epoch',self.sampling_epoch)==self.sampling_epoch
                and evidence.get('confirmed') and pid==clip['person']
                and all(s['person_id']==pid for s in ranges)):
            ranges=[dict(start_s=0.,end_s=end-start,person_id=pid,epoch=self.sampling_epoch,
                visual_stamp=clip['start'],proof='verified_start_end')]
        self.speaker.offer(dict(clip,sampling_ranges=ranges,sampling_conflict=conflict))

    def poll(self,state,session):
        now=time.monotonic()
        if now-self.last_poll<.1:return
        self.last_poll=now
        # Completed old responses must not circulate back through a bounded
        # cache forever in a long conversation. Keep recent + actually active.
        selected=dict(list(state.responses.items())[-256:])
        for rid in state.player.response_ids(getattr(state,'generating',None)):
            if rid in state.responses:selected[rid]=state.responses[rid]
        for rid,r in selected.items():
            source=None if r.get('archive_no_user') else (r.get('user_turn') or {}).get('item_id') or r.get('archive_item')
            consumed=state.player.consumed.get(rid,0)
            expected=round(r.get('received_audio_s',0)*48000)
            outcome='generating'
            if rid in state.blocked or rid in state.player.cancelled:outcome='interrupted'
            elif r.get('status') in ('failed','cancelled'):outcome=r['status']
            elif (r.get('status')=='completed' and expected and consumed>=expected
                    and r.get('empty_dac',float('inf'))<=now):outcome='complete'
            elif r.get('status')=='completed' and not expected:outcome='text_only'
            elif consumed:outcome='playing'
            reply_session=r.get('archive_session') or session
            value=(source,r.get('transcript',''),consumed,outcome)
            snapshot=(reply_session,*value)
            if self.responses.get(rid)!=snapshot:
                self.responses[rid]=snapshot;self.submit('reply',rid,reply_session,*value)
        while len(self.responses)>512:self.responses.pop(next(iter(self.responses)))

    def writer(self,relative,rate):
        path=self.store.directory/relative;path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        import os
        fd=os.open(path,os.O_CREAT|os.O_WRONLY|os.O_TRUNC,0o600)
        stream=os.fdopen(fd,'wb');writer=wave.open(stream,'wb');writer.setparams((1,2,rate,0,'NONE','not compressed'))
        return writer,stream

    def work(self):
        while True:
            try:kind,args=self.queue.get(timeout=.1)
            except queue.Empty:
                self.cleanup_outputs();continue
            if kind=='shutdown':break
            try:getattr(self,'on_'+kind)(*args)
            except Exception as exc:
                if kind=='final' and args[0] in self.source_keys:
                    self.finalized[self.source_keys[args[0]]]=dict(saved=False,status='storage_error')
                self.notices.put('记忆记录未完成：'+type(exc).__name__+'；聊天和留言账本不受此异常影响。')
            finally:
                self.queue.task_done()
                if self.queue.empty():
                    self.dropped=False;self.cleanup_outputs()
        for clip in list(self.active.values()):
            clip['writer'].close();clip['stream'].close()
        for rid in list(self.output):self.finish_output(rid)

    def finish_output(self,rid):
        out=self.output.pop(rid);out['writer'].close();out['stream'].close()
        expected=out.get('expected',out['end'])
        self.store.reply(rid,'',samples=max(expected,out['end']),recorded_samples=out['samples'],
            audio_status='gapped' if out['gapped'] or out['end']<expected else 'recorded')

    def cleanup_outputs(self):
        for rid,out in list(self.output.items()):
            if out.get('terminal') and time.monotonic()-out['last_write']>.3:self.finish_output(rid)

    def on_capture(self,stamp,pcm,far):
        self.ring.append((stamp,pcm,far))
        for clip in self.active.values():
            if clip['end'] is not None and stamp>=clip['end']:continue
            self.append_input(clip,stamp,pcm,far)

    def append_input(self,clip,stamp,pcm,far):
        begin=max(0,round((clip['start']-stamp)*16000));end=len(pcm)//2
        if clip['end'] is not None:end=min(end,max(0,round((clip['end']-stamp)*16000)))
        if end<=begin:return
        if clip['last'] is not None and stamp-clip['last']>.035:clip['alignment']='gapped'
        clip['writer'].writeframesraw(pcm[2*begin:2*end]);clip['last']=stamp+(end/16000)
        clip['near'].append(pcm[2*begin:2*end]);clip['far'].append(far[2*begin:2*end])
        # Speaker observation uses at most the last 12 seconds, not an unbounded duplicate.
        while sum(map(len,clip['near']))>384000:clip['near'].popleft();clip['far'].popleft()

    def on_start(self,item,session,person,stamp,evidence):
        key=(session,item)
        self.source_keys[item]=key
        while len(self.source_keys)>256:self.source_keys.pop(next(iter(self.source_keys)))
        if key in self.turns:return
        rid,notice=self.store.begin(session,item,person)
        self.admissions[key]=rid is not None
        if not rid:self.finalized[key]=dict(saved=False,status='quota_paused')
        while len(self.finalized)>256:self.finalized.pop(next(iter(self.finalized)))
        from vivis.interaction.continuity import actor
        if notice:
            if actor(person) not in self.blocked_people:self.notices.put(notice)
            self.blocked_people.add(actor(person))
        else:self.blocked_people.discard(actor(person))
        while len(self.admissions)>256:self.admissions.pop(next(iter(self.admissions)))
        if not rid:return
        for previous in list(self.active):self.on_end(previous,stamp,True,None)
        self.turns[key]=rid;self.live[item]=(session,rid)
        # New input closes an unanswered previous round; later linked replies may
        # still update its playback status without changing its original actor.
        if self.previous_round and self.previous_round!=rid:self.store.close(self.previous_round,'next_user_turn')
        self.previous_round=rid
        if stamp is None:
            self.store.input_audio(rid,None,'server_offset_unavailable');return
        relative='audio/'+rid+'/user.wav';writer,stream=self.writer(relative,16000)
        alignment='aligned' if self.ring and self.ring[0][0]<=stamp else 'missing_preroll'
        clip=dict(rid=rid,path=relative,writer=writer,stream=stream,start=stamp,end=None,
            last=None,alignment=alignment,near=deque(),far=deque(),person=person,evidence=evidence,session=session,item=item,
            drop_generation=self.drop_generation,sampling_epoch=self.sampling_epoch,audio_epoch=self.audio_epoch)
        self.active[item]=clip
        for frame in self.ring:self.append_input(clip,*frame)
        self.store.input_audio(rid,relative,'recording')
        while len(self.turns)>256:self.turns.pop(next(iter(self.turns)))
        while len(self.live)>256:self.live.pop(next(iter(self.live)))

    def on_final(self,item,text,accepted=True):
        entry=self.live.get(item)
        if not accepted:
            self.finished_inputs.pop(item,None);self.invalid_inputs.pop(item,None)
            if entry:self.store.discard(entry[1]);self.live.pop(item,None)
            if item in self.source_keys:self.admissions[self.source_keys[item]]=False
            return
        if not isinstance(text,str) or not text.strip():
            if entry:self.store.discard(entry[1]);self.live.pop(item,None)
            return
        recovered=self.invalid_inputs.pop(item,None)
        if not entry and recovered:
            rid,notice=self.store.begin(recovered['session'],item,recovered['person'])
            if notice:self.notices.put(notice)
            if rid:
                entry=(recovered['session'],rid);self.live[item]=entry
                path='audio/'+rid+'/user.wav';w,f=self.writer(path,16000)
                w.writeframes(recovered['pcm']);w.close();f.close()
                self.store.input_audio(rid,path,'recovered_invalid_turn_partial')
        if entry:
            self.store.final(entry[1],text)
            self.finalized[(entry[0],item)]=dict(saved=True,status='committed',round_id=entry[1])
            while len(self.finalized)>256:self.finalized.pop(next(iter(self.finalized)))
            clip=self.finished_inputs.pop(item,None)
            if clip and self.speaker:self.offer_speaker(clip)

    def on_job_notice(self,jid):
        with self.store.db() as db:db.execute('UPDATE jobs SET notice=1 WHERE id=?',(jid,))

    def on_settle(self):
        from vivis.interaction.continuity import utc
        with self.store.db() as db:
            db.execute("""UPDATE rounds SET closed=?,outcome=CASE
                WHEN NOT EXISTS(SELECT 1 FROM replies WHERE round_id=rounds.id) THEN 'unanswered'
                WHEN EXISTS(SELECT 1 FROM replies WHERE round_id=rounds.id AND outcome NOT IN ('complete','text_only')) THEN 'interrupted'
                ELSE 'complete' END
                WHERE closed IS NULL AND user_text<>'' AND NOT EXISTS(
                    SELECT 1 FROM replies WHERE round_id=rounds.id AND outcome IN ('playing','generating'))""",(utc(),))

    def on_end(self,item,stamp,valid,evidence=None):
        clip=self.active.pop(item,None)
        if not clip:return
        clip['end']=stamp
        clip['writer'].close();clip['stream'].close()
        # speech_stopped can arrive late; trim actual WAV by the server endpoint.
        if stamp is not None and stamp>=clip['start']:
            path=self.store.directory/clip['path']
            with wave.open(str(path),'rb') as r:pcm=r.readframes(round((stamp-clip['start'])*16000))
            writer,stream=self.writer(clip['path'],16000);writer.writeframes(pcm);writer.close();stream.close()
        else:clip['alignment']='end_offset_unavailable'
        if not valid:
            self.invalidate_identity_audio()
            self.invalid_inputs[item]=dict(session=clip['session'],person=clip['person'],pcm=b''.join(clip['near'])[-384000:])
            while len(self.invalid_inputs)>16:self.invalid_inputs.pop(next(iter(self.invalid_inputs)))
            self.store.discard(clip['rid']);self.live.pop(item,None);return
        if not evidence or evidence.get('person_id')!=(clip.get('evidence') or {}).get('person_id') or not evidence.get('confirmed'):
            if clip.get('evidence'):clip['evidence']['confirmed']=False
        if self.drop_generation!=clip['drop_generation']:
            clip['alignment']='recording_queue_gap';self.invalidate_identity_audio()
        self.store.input_audio(clip['rid'],clip['path'],clip['alignment'])
        if self.speaker and clip['alignment']=='aligned':
            # Remove any audio captured after the late server endpoint, retaining
            # the correctly bounded original clip for voice observation.
            if stamp is not None:
                with wave.open(str(self.store.directory/clip['path']),'rb') as r:
                    r.setpos(max(0,r.getnframes()-192000));clip['near']=deque([r.readframes(192000)])
                reference=b''.join(clip['far'])
                tail=max(0,round(((clip['last'] or stamp)-stamp)*16000))*2
                end=max(0,len(reference)-tail);wanted=len(clip['near'][0])
                clip['far']=deque([reference[max(0,end-wanted):end]])
                clip['speaker_start']=stamp-len(clip['near'][0])/32000
            if self.finalized.get((clip['session'],item),{}).get('saved'):
                self.offer_speaker(clip)
            else:
                self.finished_inputs[item]=clip
                while len(self.finished_inputs)>16:self.finished_inputs.pop(next(iter(self.finished_inputs)))

    def on_played(self,rid,values,stamp,position):
        first=rid not in self.output
        if rid not in self.output:
            relative='audio/replies/'+uuid.uuid4().hex+'.wav';w,f=self.writer(relative,48000)
            self.output[rid]=dict(writer=w,stream=f,path=relative,samples=0,end=0,gapped=False,last_write=time.monotonic())
        out=self.output[rid]
        out['gapped']|=position!=out['end'];out['end']=position+len(values)
        out['writer'].writeframesraw(np.rint(np.clip(values,-1,1)*32767).astype('<i2').tobytes())
        out['samples']+=len(values);out['last_write']=time.monotonic()
        if first:self.store.reply(rid,'',audio=out['path'],samples=out['samples'],outcome='playing',audio_status='recording')

    def on_reply(self,rid,session,item,text,samples,outcome):
        entry=self.live.get(item);round_id=entry[1] if entry and entry[0]==session else None
        self.store.reply(rid,session,round_id,text,samples,outcome=outcome)
        if outcome in ('complete','interrupted','failed','cancelled','text_only'):
            output=self.output.get(rid)
            if output:output['terminal']=True;output['expected']=samples
            if round_id is None:self.store.prune_events()
            else:
                with self.store.db() as db:
                    active=db.execute("SELECT 1 FROM replies WHERE round_id=? AND outcome IN ('playing','generating')",(round_id,)).fetchone()
                if not active:self.store.close(round_id,outcome)

    def on_disconnect(self):
        for item in list(self.active):self.on_end(item,None,True,None)
        from vivis.interaction.continuity import utc
        with self.store.db() as db:
            empty=[dict(r) for r in db.execute("SELECT id,audio FROM rounds WHERE user_text=''")]
            db.execute("DELETE FROM rounds WHERE user_text=''")
            db.execute("UPDATE rounds SET closed=?,outcome='connection_ended' WHERE closed IS NULL",(utc(),))
        for row in empty:
            if row['audio']:self.store.remove_audio(row['audio'])
        self.previous_round=None

    def close(self):
        if self.closed:return
        self.closed=True
        self.sampling_job.close()
        if self.speaker:self.speaker.job.close()
        # This is called after audio callbacks stop. Bound shutdown even if disk stalls.
        try:self.queue.put(('shutdown',()),timeout=.2)
        except queue.Full:return
        self.thread.join(timeout=2)
        if not self.thread.is_alive():
            fcntl.flock(self.owner_fd,fcntl.LOCK_UN);os.close(self.owner_fd)
