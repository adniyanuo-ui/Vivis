"""Optional realtime-trial front ends. Never gate the microphone on playback."""
from copy import deepcopy
import numpy as np
from scipy.signal import firwin,lfilter

from vivis.speech.echo import EchoCanceller

AEC_PROFILES=('baseline','voice-preserve')


def select_frontend(config,backend=None,profile=None):
    """Run-only selection; never mutate the caller's canonical configuration."""
    selected=deepcopy(config)
    if backend is not None:selected['backend']=backend
    if profile is not None:
        selected.setdefault('aec3',{})['profile']=profile
        if profile=='voice-preserve' and backend is None:selected['backend']='aec3'
    validate_frontend(selected)
    return selected


def validate_frontend(config):
    backend=config.get('backend','speex')
    if backend not in ('speex','aec3'):raise ValueError('Unknown voice AEC backend: '+str(backend))
    options=config.get('aec3',{})
    profile=options.get('profile','baseline')
    if profile not in AEC_PROFILES:raise ValueError('Unknown AEC profile: '+str(profile))
    if profile=='voice-preserve' and backend!='aec3':
        raise ValueError('voice-preserve requires the aec3 backend')
    if type(options.get('ns_level',1)) is not int or not 0<=options.get('ns_level',1)<=3:
        raise ValueError('AEC3 ns_level must be0..3')
    delay=options.get('stream_delay_ms',0)
    if type(delay) is not int or not 0<=delay<=500:raise ValueError('AEC3 delay hint must be0..500ms')
    return backend


class Aec3:
    """One processor per DSP thread; exact20ms input becomes two10ms blocks."""
    def __init__(self,options):
        self.profile=options.get('profile','baseline')
        validate_frontend(dict(backend='aec3',aec3=options))
        extra={}
        try:
            if self.profile=='voice-preserve':
                import _vivis_aec3
                if getattr(_vivis_aec3,'vivis_profile_api_version',None)!=1:
                    raise RuntimeError('Unexpected voice-preserve API; rebuild with ./tools/build_voice_preserve.sh')
                AudioProcessor=_vivis_aec3.AudioProcessor
                extra['nearend_threshold']=1.0
            else:
                from pywebrtc_audio import AudioProcessor
        except ImportError as exc:
            command='./tools/setup_realtime_voice.sh'+(' --voice-preserve' if self.profile=='voice-preserve' else '')
            raise RuntimeError('AEC3 dependency missing; run '+command) from exc
        self.engine=AudioProcessor(sample_rate=16000,num_channels=1,echo_cancellation=True,
            noise_suppression=True,ns_level=options.get('ns_level',1),auto_gain_control=False,
            stream_delay_ms=options.get('stream_delay_ms',0),**extra)

    def process(self,near,far):
        if near.shape!=(320,) or far.shape!=(320,):raise ValueError('AEC3 needs complete20ms mono frames')
        # No zero padding or resets at network packet / response boundaries.
        return self.engine.process(np.ascontiguousarray(near,dtype=np.float32),
                                   np.ascontiguousarray(far,dtype=np.float32))

    def close(self):self.engine=None


class VoiceFrontEnd:
    """Shared48k->16k filtering and backend, used by live I/O and offline replay."""
    def __init__(self,speech,options=None,enabled=True):
        options=options or {};self.backend=validate_frontend(options) if enabled else 'off'
        if speech['aec']['sample_rate_hz']!=16000:raise ValueError('Voice front end requires16kHz processing')
        self.coefficients=firwin(63,1/3-.015);self.near_zi=np.zeros(62);self.far_zi=np.zeros(62)
        self.echo=(Aec3(options.get('aec3',{})) if self.backend=='aec3'
                   else EchoCanceller(speech['aec']) if self.backend=='speex' else None)

    def process16(self,near,far):
        if near.shape!=(320,) or far.shape!=(320,) or not np.isfinite(near).all() or not np.isfinite(far).all():
            raise ValueError('Voice front end requires finite20ms mono16k frames')
        out=self.echo.process(near,far) if self.echo else near.copy()
        if out.shape!=(320,) or not np.isfinite(out).all():raise RuntimeError('Invalid voice front-end output')
        return out

    def process48(self,near,far):
        if near.shape!=(960,) or far.shape!=(960,):raise ValueError('Voice front end requires20ms mono48k frames')
        near,self.near_zi=lfilter(self.coefficients,1,near,zi=self.near_zi)
        far,self.far_zi=lfilter(self.coefficients,1,far,zi=self.far_zi)
        near=near[::3].astype(np.float32);far=far[::3].astype(np.float32)
        return near,far,self.process16(near,far)

    def close(self):
        if self.echo:self.echo.close();self.echo=None
