"""Deterministic committed-utterance routing; no LLM in the stop path."""
import re
import unicodedata


def normalize(text):
    return re.sub(r'[\s，。！？、,.!?：:；;"“”‘’\-]', '',unicodedata.normalize('NFKC',text).lower())


def command_text(text):
    value=normalize(text)
    for _ in range(3):value=re.sub(r'^(?:小帅|机器人|请你|请|麻烦你|麻烦|帮我|你)', '',value)
    return re.sub(r'(?:好吗|好不好|可以吗|吧|呀|啊)$','',value)


class IntentRouter:
    def __init__(self,catalog):
        if catalog.get('schema_version')!='vivis-programs-1.0':raise ValueError('Unsupported program catalog')
        self.programs=catalog['programs'];self.stop={command_text(s) for s in catalog['standby_phrases']}
        if len({p['id'] for p in self.programs})!=len(self.programs):raise ValueError('Repeated program IDs')

    def parse(self,text,active=False,current_program=None):
        if not isinstance(text,str) or len(text)>256:return dict(kind='ignore')
        value=command_text(text)
        # Remove polite addressing, never arbitrary words inside an utterance.
        if value in self.stop or re.fullmatch(r'(?:停止|结束|退出)(?:(?:当前|这个|本次)?(?:节目|播放|表演|跟随|唱歌|歌曲|模仿|学习))?',value):
            return dict(kind='standby',reason=text)
        if value.startswith(('不要','不用','别','不想','不是','停止是什么意思','怎么停止')):
            return dict(kind='ignore')
        if value in ('你是谁','是谁','叫什么','你叫什么','介绍一下自己'):
            return dict(kind='prompt',name='identity')
        if value in ('有哪些节目','有什么节目','节目单','你会什么','会什么','能做什么','你能做什么','帮助'):
            return dict(kind='prompt',name='menu')
        matches=[p['id'] for p in self.programs if any(normalize(a) in value for a in p['aliases'])]
        if len(matches)>1:return dict(kind='prompt',name='clarify')
        if not matches:
            return dict(kind='prompt',name='clarify') if value.startswith(('播放','开始','唱','切换','换成','表演')) else dict(kind='ignore')
        selected=next(p for p in self.programs if p['id']==matches[0])
        aliases={normalize(a) for a in selected['aliases']}
        if active and value in aliases and (current_program is None or current_program==selected['id']):
            return dict(kind='ignore')  # Its own title in program audio is not a restart request.
        residual=value
        for alias in sorted(aliases,key=len,reverse=True):residual=residual.replace(alias,'')
        if residual and residual not in ('播放','开始','唱','唱一首','来一首','来首','来个','表演','表演一下','我要听',
            '我想听','我想看','想听','想看','我要看','切换到','切换','换成','换到','给我唱','给我播放','请播放'):
            return dict(kind='ignore')
        return dict(kind='select',program=selected['id'])


def resolve_candidates(router,final,streaming):
    """Choose among two COMPLETE hypotheses; never invent a misspelling alias."""
    primary=router.parse(final);alternative=router.parse(streaming)
    negative=bool(re.match(r'^(?:我)?(?:不要|不用|别|不想|不是)',command_text(final))) and primary['kind']=='ignore'
    if negative:return final,'sensevoice_final'
    valid={'standby','select'}
    if primary['kind'] in valid and alternative['kind'] in valid:
        if primary['kind']=='standby':return final,'sensevoice_final'
        if alternative['kind']=='standby':return streaming,'streaming_final_stop'
        if primary['program']!=alternative['program']:return final+'或者'+streaming,'ambiguous_final'
    uncertain=primary['kind']=='ignore' or primary==dict(kind='prompt',name='clarify')
    if uncertain and alternative['kind'] in valid:return streaming,'streaming_final_fallback'
    return final,'sensevoice_final'
