"""Map a PortAudio stream clock without injecting Python callback scheduling jitter."""
from dataclasses import dataclass,asdict
import math
import time


@dataclass(frozen=True)
class AudioStreamClock:
    offset_s: float
    calibration_roundtrip_s: float
    stream_time_s: float

    @classmethod
    def calibrate(cls,stream,clock=None,samples=9):
        """Call from the owning thread while OPEN but before START, never in a callback.

        Pa_GetStreamTime is valid for an open stopped stream and shares its
        callback clock. Choose the shortest measured call to limit query skew.
        The roundtrip is a sampling diagnostic, not a hardware accuracy claim.
        """
        clock=clock or time.monotonic;observations=[]
        for _ in range(samples):
            before=clock();pa=float(stream.time);after=clock()
            if not all(math.isfinite(v) for v in (before,pa,after)) or pa==0 or after<before:
                raise RuntimeError('Invalid PortAudio stream clock; cannot align the echo reference')
            observations.append((after-before,(before+after)/2-pa,pa))
        if not observations:raise ValueError('Clock calibration needs at least one sample')
        elapsed,offset,pa=min(observations)
        return cls(offset,elapsed,pa)

    def convert(self,stream_time):
        value=float(stream_time)+self.offset_s
        if not math.isfinite(value):raise RuntimeError('Nonfinite audio stream timestamp')
        return value

    def metadata(self):return dict(method='Pa_GetStreamTime_fixed_offset',**asdict(self))
