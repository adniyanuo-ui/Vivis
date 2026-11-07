"""Pure three-mode session policy; no devices, processes or model inference."""
STANDBY='standby'
INTERACTION='interaction'


class SessionPolicy:
    def __init__(self,program_ids):
        self.programs=set(program_ids)
        if not self.programs or self.programs&{STANDBY,INTERACTION}:raise ValueError('Invalid/reserved program IDs')
        self.desired=STANDBY;self.pending_prompt=None

    def mode_of(self,target):
        if target in (STANDBY,INTERACTION):return target
        if target in self.programs:return 'task'
        raise ValueError('Unknown session target')

    @property
    def mode(self):return self.mode_of(self.desired)

    def request(self,intent):
        """Return whether the current child needs cancellation/restart."""
        kind=intent['kind'];previous=self.desired
        if self.mode==STANDBY and kind!='wake':return False
        if kind=='wake':
            if self.mode==STANDBY:self.desired=INTERACTION;self.pending_prompt='welcome'
            followup=intent.get('followup')
            restart=self.request(followup) if followup is not None else False
            return self.desired!=previous or restart
        if kind=='select':
            if intent['program'] not in self.programs:raise ValueError('Unknown selected program')
            self.desired=intent['program'];self.pending_prompt=None
            return True
        if kind=='standby':
            self.desired=STANDBY;self.pending_prompt='standby'
        elif kind=='interaction':
            self.desired=INTERACTION
            self.pending_prompt='interaction' if previous!=INTERACTION else None
        elif kind=='prompt' and self.mode==INTERACTION:self.pending_prompt=intent['name']
        return self.desired!=previous

    def completed(self,target,cancelled=False):
        if not cancelled and self.desired==target and self.mode_of(target)=='task':
            self.desired=INTERACTION;self.pending_prompt='interaction'

    def take_prompt(self,target,ready):
        if ready and target==self.desired:
            value=self.pending_prompt;self.pending_prompt=None;return value
        return None
