"""Optional proactive welcomes yield to user evidence before first playback."""


class WelcomeTurnGate:
    def __init__(self):
        self.user_seen=False;self.active=False;self.pending=False;self.cancelled=False;self.responses=set()

    def arm(self):
        if self.user_seen:return False
        self.active=True;self.cancelled=False;self.responses.clear();return True

    def sent(self):
        if self.active:self.pending=True

    def created(self,rid):
        if not self.pending:return False
        self.pending=False;self.responses.add(rid)
        return self.cancelled

    def user_event(self,event,player):
        # Even an empty ASR delta establishes an input segment. It grants no
        # identity, business authority or right to interrupt ordinary speech.
        if event.get('type') not in ('input_audio_buffer.speech_started',
                'conversation.item.input_audio_transcription.delta',
                'conversation.item.input_audio_transcription.completed') or not event.get('item_id'):return False
        self.user_seen=True
        if not self.active:return False
        self.active=False
        if player.has_started(self.responses):return False
        self.cancelled=True;return True
