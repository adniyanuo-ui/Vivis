"""Final-ASR wake admission and server-confirmed audio permission."""
import asyncio
import json
import re

from vivis.speech.runtime_policy import policy,VoiceTransportError,bounded_send


def addressed_wake(text,words):
    if not isinstance(text,str) or not 1<=len(text)<=1000:return False
    value=re.sub(r'[\s，,。!！?？、]+','',text)
    for word in words:
        if not isinstance(word,str) or not word:continue
        match=re.match(r'^(?:喂|嘿|哎|你好|您好)?'+re.escape(word),value)
        if not match:continue
        tail=value[match.end():]
        # Mentioning the name in a report/quotation is not addressing the robot.
        if tail.startswith(('说','告诉','的','这个名字','刚才','昨天','今天说','明天')):continue
        return True
    return False


class AudioPermissionGate:
    """Only session.updated with audio enabled releases an explicit wake reply."""
    def __init__(self):self.pending=None

    async def enable(self,ws,instructions):
        future=asyncio.get_running_loop().create_future();self.pending=future
        try:
            await bounded_send(ws,json.dumps(dict(type='session.update',session=dict(
                instructions=instructions,modalities=['text','audio'])),ensure_ascii=False))
            try:await asyncio.wait_for(future,policy()['controller_ack_s'])
            except asyncio.TimeoutError:raise VoiceTransportError('Wake audio permission acknowledgement timed out') from None
        finally:self.pending=None

    def observe(self,event):
        if (event.get('type')=='session.updated' and 'audio' in event.get('session',{}).get('modalities',[])
                and self.pending is not None and not self.pending.done()):self.pending.set_result(True)
