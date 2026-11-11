"""Narrow, bounded recovery for explicit transient Qwen service errors."""
from collections import deque
import math
import time


class TransientServiceError(RuntimeError):
    def __init__(self,code,message,event_id=None):
        self.code=code;self.event_id=event_id
        super().__init__(code+': '+message)


def transient_error(event):
    if event.get('type')=='error':error=event.get('error',{})
    elif event.get('type')=='response.done' and event.get('response',{}).get('status')=='failed':
        details=event['response'].get('status_details')
        error=details.get('error',{}) if isinstance(details,dict) else {}
    else:return None
    if isinstance(error,dict) and error.get('code') in ('InternalError','ServiceUnavailable'):
        return TransientServiceError(error['code'],str(error.get('message',''))[:1000],event.get('event_id'))
    return None


class ServiceRecovery:
    def __init__(self,config):
        limit=config.get('max_reconnects');window=config.get('window_s');delays=config.get('backoff_s')
        if (type(limit) is not int or not 1<=limit<=5
                or type(window) not in (int,float) or not math.isfinite(window) or not 10<=window<=300
                or not isinstance(delays,list) or len(delays)!=limit
                or any(type(d) not in (int,float) or not math.isfinite(d) or not 0<d<=10 for d in delays)):
            raise ValueError('Invalid transient service recovery policy')
        self.limit=limit;self.window=window;self.delays=tuple(delays);self.attempts=deque()

    def next(self,now=None):
        now=time.monotonic() if now is None else now
        while self.attempts and now-self.attempts[0]>=self.window:self.attempts.popleft()
        if len(self.attempts)>=self.limit:raise RuntimeError('Qwen transient recovery exhausted within '+str(self.window)+' seconds')
        self.attempts.append(now)
        return dict(attempt=len(self.attempts),maximum=self.limit,delay_s=self.delays[len(self.attempts)-1],window_s=self.window)


def transport_error(exc):
    from vivis.speech.runtime_policy import VoiceTransportError
    import errno
    import socket
    if isinstance(exc,(VoiceTransportError,TimeoutError,ConnectionError,socket.gaierror)):return True
    if isinstance(exc,OSError) and exc.errno in (errno.ENETUNREACH,errno.ENETDOWN,errno.EHOSTUNREACH,errno.ETIMEDOUT):return True
    try:
        from websockets.exceptions import ConnectionClosed,InvalidStatus
        if isinstance(exc,ConnectionClosed):return True
        if isinstance(exc,InvalidStatus):return exc.response.status_code in (408,429,500,502,503,504)
    except ImportError:pass
    return False
