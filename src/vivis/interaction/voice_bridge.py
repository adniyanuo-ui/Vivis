"""Supervisor-side speaker handoff; stale epochs cannot regain control."""
import json
import time


class VoiceBridge:
    def __init__(self,stream):
        self.stream=stream;self.epoch=0;self.target=False;self.observed_epoch=0
        self.status='off';self.owned=False;self.failed=False;self.release_requested_at=None;self.last_program=None;self.context_revision=0;self.last_context=None
        self.transition_reply_id=None
        self.transition_reply_target='standby'
        self.preparation_speech=False

    @property
    def released(self):
        return not self.target and not self.owned and self.observed_epoch==self.epoch

    def send(self,**message):
        self.stream.write(json.dumps(message,ensure_ascii=False)+'\n');self.stream.flush()

    def enable(self,value):
        if not value:self.failed=False
        if value and self.failed:return
        if self.target==value:return
        self.target=value;self.epoch+=1
        self.release_requested_at=None if value else time.monotonic()
        self.send(kind='voice_mode',enabled=value,epoch=self.epoch,
                  **({'last_program':dict(self.last_program)} if value and self.last_program else {}))

    def context(self,mode,target,ready,*,speech_ready=None,standby_fault=False):
        reply=self.transition_reply_id if target==self.transition_reply_target else None
        early=bool(speech_ready) and not ready and (mode=='interaction' or reply is not None and not standby_fault)
        value=(mode,target,bool(ready),reply,early,bool(standby_fault))
        if value==self.last_context:return
        self.last_context=value;self.context_revision+=1
        self.send(kind='voice_context',revision=self.context_revision,mode=mode,target=target,ready=bool(ready),
                  **({'speech_ready':True} if early else {}),
                  **({'standby_fault':True} if standby_fault and mode=='standby' else {}),
                  **({'transition_reply_id':reply} if reply else {}))

    def interrupt(self):self.send(kind='voice_interrupt',epoch=self.epoch)

    def accept(self,event):
        if event.get('kind')=='voice_reply_done':
            if event.get('reply_id')!=self.transition_reply_id or self.transition_reply_id is None:return False
            self.transition_reply_id=None;return True
        if event.get('epoch')!=self.epoch:return False
        if event.get('kind')=='voice_state':
            self.observed_epoch=self.epoch;self.status=event['status'];self.owned=event['speaker_owned']
            if self.status=='unavailable':self.failed=True;self.target=False
        return True
