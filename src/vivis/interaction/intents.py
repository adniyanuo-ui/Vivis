"""Deterministic committed-utterance routing; no LLM in the stop path."""
import re
import unicodedata


def normalize(text):
    return re.sub(r'[\s，。！？、,.!?：:；;"“”‘’\-~～…]', '',unicodedata.normalize('NFKC',text).lower())


def command_text(text):
    value=normalize(text)
    for _ in range(4):value=re.sub(r'^(?:嗯|呃|小帅|机器人|请你|请|麻烦你|麻烦|帮我|你们|你)', '',value)
    return re.sub(r'(?:好吗|好不好|可以吗|吧|呀|啊|哈|哟|哦|噢|啦|喽|咯|呗|呢|了)+$','',value)


class IntentRouter:
    def __init__(self,catalog,wake_words=None):
        if catalog.get('schema_version')!='vivis-programs-1.0':raise ValueError('Unsupported program catalog')
        self.programs=catalog['programs'];self.sleep={command_text(s) for s in catalog['standby_phrases']}
        self.stop={command_text(s) for s in catalog.get('exit_phrases',[])}
        questions=catalog.get('questions',{})
        self.identity={command_text(s) for s in questions.get('identity',['你是谁','叫什么','你叫什么','介绍一下自己'])}
        self.menu={command_text(s) for s in questions.get('menu',['有哪些节目','有什么节目','节目单','你会什么','能做什么','能帮我做什么','帮助'])}
        self.observed={command_text(k):v for k,v in catalog.get('observed_asr_commands',{}).items()}
        if any(v.get('kind')!='select' or v.get('program') not in {p['id'] for p in self.programs}
               for v in self.observed.values()):raise ValueError('Invalid observed ASR command')
        self.wake_words=wake_words if wake_words is not None else catalog.get('wake_words',['小帅'])
        if (not isinstance(self.wake_words,list) or not self.wake_words
                or any(not isinstance(w,str) or not 1<=len(normalize(w))<=20 for w in self.wake_words)):
            raise ValueError('Need configured wake words')
        if len({p['id'] for p in self.programs})!=len(self.programs):raise ValueError('Repeated program IDs')

    def parse(self,text,active=False,current_program=None,state=None):
        if not isinstance(text,str) or len(text)>256:return dict(kind='ignore')
        if state not in (None,'standby','interaction','task'):raise ValueError('Unknown dialogue mode')
        raw=normalize(text)
        raw=re.sub(r'^(?:你好|您好|喂)','',raw)
        wake=next((normalize(w) for w in sorted(self.wake_words,key=len,reverse=True) if raw.startswith(normalize(w))),None)
        if wake is not None:
            rest=raw[len(wake):]
            if command_text(rest) in ('','你好','您好'):
                return dict(kind='wake') if state in (None,'standby') else dict(kind='prompt',name='welcome') if state=='interaction' else dict(kind='ignore')
            followup=self._command(rest,active,current_program)
            # Do not turn quoted discussion/negation after a name into a wake.
            if followup['kind']=='ignore':return followup
            return dict(kind='wake',followup=followup) if state in (None,'standby') else followup
        if state=='standby':return dict(kind='ignore')
        return self._command(text,active,current_program)

    def _command(self,text,active=False,current_program=None):
        value=command_text(text)
        # Remove polite addressing, never arbitrary words inside an utterance.
        if value in self.sleep:return dict(kind='standby',reason=text)
        if value in self.stop or re.fullmatch(r'(?:停止|结束|退出)(?:(?:当前|这个|本次)?(?:节目|任务|播放|表演|跟随|唱歌|歌曲|音乐|曲子|朗诵|剧本|模仿|学习))?(?:一下)?',value):
            return dict(kind='interaction',reason=text)
        if value.startswith(('不要','不用','别','不想','不是','停止是什么意思','怎么停止')):
            return dict(kind='ignore')
        if value in self.identity:
            return dict(kind='prompt',name='identity')
        if value in self.menu:
            return dict(kind='prompt',name='menu')
        if value in self.observed:return dict(self.observed[value])
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
    valid={'standby','interaction','select','wake'}
    if primary['kind'] in valid and alternative['kind'] in valid:
        if primary['kind'] in ('standby','interaction'):return final,'sensevoice_final'
        if alternative['kind'] in ('standby','interaction'):return streaming,'streaming_final_stop'
        if primary['kind']=='wake':return final,'sensevoice_final'
        if alternative['kind']=='wake':return streaming,'streaming_final_wake'
        if primary['program']!=alternative['program']:return final+'或者'+streaming,'ambiguous_final'
    uncertain=primary['kind']=='ignore' or primary==dict(kind='prompt',name='clarify')
    if alternative['kind']=='wake' and alternative.get('followup')==primary:return streaming,'streaming_final_wake'
    if uncertain and alternative['kind'] in valid:return streaming,'streaming_final_fallback'
    return final,'sensevoice_final'
