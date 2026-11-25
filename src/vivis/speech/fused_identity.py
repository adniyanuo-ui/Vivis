"""Bind local image/voice evidence to a visit; keep accepted turns immutable."""
import time
import uuid

from vivis.interaction.identity_fusion import IdentityFusion,configuration,fresh,unavailable,evidence_label
from vivis.paths import ROOT


class FusedIdentityVoice:
    def init_fused_identity(self):
        config=configuration(self.speech['managed_voice'].get('identity_fusion_config'),ROOT)
        self.identity_fusion=IdentityFusion(config) if config and config['enabled'] and self.continuity else None
        self.voice_binding=None;self.voice_context=None;self.voice_display=None;self.voice_pending=None
        self.voice_business=unavailable();self.fusion_last=None
        if self.identity_fusion and self.continuity.speaker:
            self.continuity.speaker.result_handler=self.accept_speaker_identity
        if self.identity_fusion:self.log.write('identity_fusion_enabled',voice_permissions=True,accepted_turns_immutable=True)

    def sync_voice_binding(self):
        binding=(self.message_connection,self.memory_session)
        if self.voice_binding!=binding:
            self.voice_binding=binding;self.voice_context=uuid.uuid4().hex
            self.voice_display=None;self.voice_pending=None;self.voice_business=unavailable('new_connection_or_visit')
            self.identity_fusion.reset_pairs()
            if self.continuity:self.continuity.submit('invalidate_identity_audio')
        return self.voice_context

    def fused_face_evidence(self,now=None):
        now=time.monotonic() if now is None else now
        snapshot=getattr(self.continuity,'visual_evidence',None) or {}
        if snapshot.get('blocked'):return unavailable('multiple_people')
        return fresh(snapshot.get('identity_evidence'),now,self.identity_fusion.c['face_max_age_s'],self.knowledge.people)

    def current_voice_evidence(self,now=None):
        now=time.monotonic() if now is None else now
        self.sync_voice_binding();result=self.voice_display
        if not result or result['item_id']!=self.turn_item_id:return unavailable('current_turn_not_ready')
        return fresh(result.get('identity_evidence'),now,self.identity_fusion.c['voice_max_age_s'],self.knowledge.people)

    def speaker_status(self,person=None):
        """Capability, gallery readiness and this turn's result are separate facts."""
        observer=getattr(getattr(self,'continuity',None),'speaker',None)
        if observer is None:return dict(enabled=False,current_result=unavailable('speaker_disabled'))
        rows=observer.gallery.get(person,[]) if person else []
        active=sum(r.get('active',True) for r in rows)
        return dict(enabled=True,library_present=bool(observer.gallery),
            current_person_references=len(rows),current_person_active_references=active,
            minimum_active_references=observer.c['enrollment_samples'],
            current_person_eligible=active>=observer.c['enrollment_samples'],
            current_result=self.current_voice_evidence() if self.identity_fusion else unavailable('observation_only'),
            notice='当前声音无结果不等于没有声纹库；保留会话身份不代表刚刚听声认出。')

    def accept_speaker_identity(self,result):
        """Called by poll on the cloud loop, never from the model's worker."""
        token=self.sync_voice_binding()
        rejected=[name for name,bad in (
            ('connection_changed',result.get('voice_context')!=token),('visit_changed',result.get('session')!=self.memory_session),
            ('item_changed',result.get('item_id')!=self.turn_item_id),('generation_changed',result.get('source_generation')!=self.input_generation),
            ('audio_invalidated',result.get('audio_epoch')!=self.continuity.audio_epoch)) if bad]
        if rejected:
            self.log.write('speaker_identity_stale',turn_id=result['turn_id'],not_applied=True,reasons=rejected);return
        self.voice_display=dict(result)
        evidence=result.get('identity_evidence') or unavailable()
        judged=self.identity_fusion.stage_voice(self.fused_face_evidence(),evidence,time.monotonic(),self.knowledge.people)
        self.voice_pending=dict(result,adjudicated_evidence=judged)
        self.log.write('speaker_identity_ready',turn_id=result['turn_id'],identity=evidence,
            applies_at_next_boundary=True,original_actor_unchanged=True,
            sampling_conflict_only=bool(result.get('sampling_conflict')))

    def admit_voice_identity(self,*,previous_turn=False):
        if not self.identity_fusion:return
        self.sync_voice_binding();result=self.voice_pending
        if not result:return
        self.voice_pending=None
        generations={self.input_generation,self.input_generation-1} if previous_turn else {self.input_generation}
        if (result.get('item_id')!=self.turn_item_id or result.get('source_generation') not in generations
                or result.get('audio_epoch')!=self.continuity.audio_epoch):return
        self.voice_business=fresh(result.get('adjudicated_evidence') or result.get('identity_evidence'),time.monotonic(),
            self.identity_fusion.c['voice_max_age_s'],self.knowledge.people)

    def finish_identity_turn_start(self):
        if self.identity_fusion:
            # The preceding voice has helped bind this new turn. It must not
            # keep voting as if it were this turn's freshly observed voice.
            self.voice_business=unavailable('new_user_turn');self.voice_display=None

    def voice_conflicts_with_actor(self,actor):
        if not self.identity_fusion or not actor or actor.startswith('anonymous:'):return False
        face=self.fused_face_evidence();voice=self.current_voice_evidence()
        # User-selected policy: a confirmed image remains the primary identity.
        if face['state']=='confirmed':return face['person_id']!=actor
        if self.voice_pending:
            judged=fresh(self.voice_pending.get('adjudicated_evidence'),time.monotonic(),
                self.identity_fusion.c['voice_max_age_s'],self.knowledge.people)
            if judged['state']=='confirmed':voice=judged
        return voice['state']=='confirmed' and voice['person_id']!=actor

    def fused_identity(self):
        self.sync_voice_binding();now=time.monotonic()
        self.fusion_last=self.identity_fusion.observe(self.fused_face_evidence(now),self.voice_business,now,self.knowledge.people)
        return self.fusion_last

    def fused_identity_labels(self,stable):
        person=self.knowledge.people.get(stable['person_id'],{})
        logical=person.get('preferred_address') or person.get('display_name') or 'other'
        image=stable['face'];voice=self.current_voice_evidence()
        image_label=evidence_label(image,self.knowledge.people,'图');voice_label=evidence_label(voice,self.knowledge.people,'声')
        return dict(identity_label=logical+'-'+image_label+'-'+voice_label,logical_label=logical,
            realtime_label=image_label,voice_label=voice_label,visual_state=image['state'],voice_state=voice['state'])
