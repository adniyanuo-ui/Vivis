"""Supervisor-side speaker handoff; stale epochs cannot regain control."""
import json
import time


class VoiceBridge:
    def __init__(self,stream):
        self.stream=stream;self.epoch=0;self.target=False;self.observed_epoch=0
        self.status='off';self.owned=False;self.failed=False;self.release_requested_at=None;self.last_program=None;self.context_revision=0;self.last_context=None

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

    def context(self,mode,target,ready):
        value=(mode,target,bool(ready))
        if value==self.last_context:return
        self.last_context=value;self.context_revision+=1
        self.send(kind='voice_context',revision=self.context_revision,mode=mode,target=target,ready=bool(ready))

    def interrupt(self):self.send(kind='voice_interrupt',epoch=self.epoch)

    def accept(self,event):
        if event.get('epoch')!=self.epoch:return False
        if event.get('kind')=='voice_state':
            self.observed_epoch=self.epoch;self.status=event['status'];self.owned=event['speaker_owned']
            if self.status=='unavailable':self.failed=True;self.target=False
        return True
