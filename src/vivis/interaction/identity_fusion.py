"""Evidence-based reception identity; missing observations are not rejections."""
import json
import math
from pathlib import Path


def configuration(path,root):
    if not path:return None
    data=json.loads((Path(root)/path).read_text())
    if data.get('schema_version')!='vivis-identity-fusion-1.0' or type(data.get('enabled')) is not bool:
        raise ValueError('Invalid identity fusion configuration')
    for key,low,high in [('face_max_age_s',.3,3),('voice_max_age_s',1,15),('pair_max_skew_s',.2,3)]:
        value=data.get(key)
        if type(value) not in (int,float) or not math.isfinite(value) or not low<=value<=high:
            raise ValueError('Invalid identity fusion duration: '+key)
    if type(data.get('pair_confirmation_samples')) is not int or not 2<=data['pair_confirmation_samples']<=4:
        raise ValueError('Invalid joint identity confirmation count')
    return data


def unavailable(reason='no_result'):
    return dict(state='unavailable',person_id=None,stamp=None,evidence_id=None,reason=reason)


def fresh(evidence,now,max_age,people):
    if not isinstance(evidence,dict):return unavailable()
    state=evidence.get('state');stamp=evidence.get('stamp');pid=evidence.get('person_id')
    if state=='unavailable':return unavailable(evidence.get('reason','no_result'))
    if (state not in ('confirmed','suspected','other') or type(stamp) not in (int,float)
            or not math.isfinite(stamp) or not 0<=now-stamp<=max_age
            or not isinstance(evidence.get('evidence_id'),str) or not evidence['evidence_id']):
        return unavailable('stale_or_invalid_result')
    if state in ('confirmed','suspected') and pid not in people:return unavailable('unknown_profile')
    return dict(evidence,person_id=None if state=='other' else pid)


def evidence_label(evidence,people,channel):
    state=evidence.get('state')
    pid=evidence.get('person_id');person=people.get(pid,{})
    name=('None' if state=='unavailable' else 'other' if state=='other' else
        person.get('preferred_address') or person.get('display_name') or 'other')
    return name+'/'+channel+('/疑' if state=='suspected' else '')


class IdentityFusion:
    def __init__(self,config):
        self.c=config;self.person=None;self.released_since=None;self.last_now=None
        self.pair_person=None;self.pairs=[];self.pair_started=None;self.reason='no_result'

    def reset(self,now=None):
        self.person=None;self.reset_pairs();self.released_since=now

    def reset_pairs(self):self.pair_person=None;self.pairs=[];self.pair_started=None

    def stage_voice(self,face,voice,now,people):
        """Pair observations while fresh, without changing the current business actor."""
        old=(self.person,self.released_since,self.reason)
        result=self.observe(face,voice,now,people)
        self.person,self.released_since,self.reason=old
        if result['reason']=='joint_confirmed':
            return dict(voice,state='confirmed',origin='joint',person_id=result['person_id'])
        return voice

    def observe(self,face,voice,now,people):
        if self.last_now is not None and now<self.last_now:self.reset(now)
        self.last_now=now
        if self.person not in people:self.person=None
        if self.pair_started is not None and now-self.pair_started>self.c['voice_max_age_s']:
            self.pairs=[];self.pair_person=None;self.pair_started=None
        face=fresh(face,now,self.c['face_max_age_s'],people)
        voice=fresh(voice,now,self.c['voice_max_age_s'],people)
        a,b=face['state'],voice['state'];old=self.person
        conflict=a==b=='confirmed' and face['person_id']!=voice['person_id']
        if a=='confirmed':self.person=face['person_id'];self.reason='image_confirmed'
        elif b=='confirmed':self.person=voice['person_id'];self.reason='joint_confirmed' if voice.get('origin')=='joint' else 'voice_confirmed'
        elif a==b=='other':self.person=None;self.reason='both_other'
        elif 'unavailable' in (a,b):self.reason='missing_evidence_hold'
        elif a==b=='suspected' and face['person_id']==voice['person_id']:
            pid=face['person_id'];self.reason='joint_confirmation_pending'
            if self.person!=pid:self.person=None
            if abs(face['stamp']-voice['stamp'])<=self.c['pair_max_skew_s']:
                if self.pair_person!=pid:self.pair_person=pid;self.pairs=[];self.pair_started=now
                ids=set(voice.get('source_turn_ids') or [voice['evidence_id']])
                if (not self.pairs or face['evidence_id']!=self.pairs[-1][0]
                        and ids.isdisjoint(set().union(*(p[1] for p in self.pairs)))):
                    self.pairs.append((face['evidence_id'],ids))
                    self.pairs=self.pairs[-self.c['pair_confirmation_samples']:]
                if len(self.pairs)>=self.c['pair_confirmation_samples']:
                    self.person=pid;self.reason='joint_confirmed'
            # Delayed admission must not destroy a pair made while its source
            # frame/voice were aligned. It cannot add a new pair either.
        else:
            candidates={e['person_id'] for e in (face,voice) if e['state']=='suspected'}
            if old not in candidates:self.person=None
            self.reason='weak_history_hold' if self.person else 'weak_evidence_rejected'
        if ('confirmed' in (a,b) or 'unavailable' not in (a,b)
                and not (a==b=='suspected' and face['person_id']==voice['person_id'])):
            self.pairs=[];self.pair_person=None;self.pair_started=None
        if old!=self.person:self.released_since=now if self.person is None else None
        confirmed=self.reason in ('image_confirmed','voice_confirmed','joint_confirmed')
        held=self.person is not None and not confirmed
        return dict(person_id=self.person,state='reviewing' if held else 'recognized' if self.person else 'unknown',
            held=held,candidate_id=face['person_id'],change_started=None,released_since=self.released_since,
            remaining_s=0.,reason=self.reason,conflict=conflict,face=face,voice=voice,policy='evidence_fusion')
