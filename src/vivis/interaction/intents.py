"""Deterministic committed-utterance routing; no LLM in the stop path."""
import re
import unicodedata


def normalize(text):
    return re.sub(r'[\s，。！？、,.!?：:；;"“”‘’《》\-~～…]', '',unicodedata.normalize('NFKC',text).lower())


def command_text(text):
    value=normalize(text)
    value=re.sub(r'^(?:好呀|好的|好啊|行啊)[,，]?', '',value)
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

    def parse(self,text,active=False,current_program=None,state=None,allow_title_correction=False,previous_program=None):
        if not isinstance(text,str) or len(text)>256:return dict(kind='ignore')
        if state not in (None,'standby','interaction','task'):raise ValueError('Unknown dialogue mode')
        raw=normalize(text)
        raw=re.sub(r'^(?:你好|您好|喂)','',raw)
        wake=next((normalize(w) for w in sorted(self.wake_words,key=len,reverse=True) if raw.startswith(normalize(w))),None)
        if wake is not None:
            rest=raw[len(wake):]
            if command_text(rest) in ('','你好','您好'):
                return dict(kind='wake') if state in (None,'standby') else dict(kind='prompt',name='welcome') if state=='interaction' else dict(kind='ignore')
            followup=self.parse(rest,active,current_program,
                state='interaction' if state in (None,'standby') else state,
                allow_title_correction=allow_title_correction,previous_program=previous_program)
            # Do not turn quoted discussion/negation after a name into a wake.
            if followup['kind']=='ignore':return followup
            return dict(kind='wake',followup=followup) if state in (None,'standby') else followup
        if state=='standby':return dict(kind='ignore')
        if state=='interaction' and command_text(text) in ('再来一遍','再来一次','再播一遍','再播一次','再播放一遍','再表演一遍','再演一遍'):
            if previous_program in {p['id'] for p in self.programs}:return dict(kind='select',program=previous_program)
            return dict(kind='prompt',name='clarify')
        result=self._command(text,active,current_program)
        if (allow_title_correction and state=='interaction'
                and (result['kind']=='ignore' or result==dict(kind='prompt',name='clarify'))):
            return self.correct_title(text) or result
        return result

    def correct_title(self,text):
        """One-substitution correction of a WHOLE requested catalog name only.

        No aliases made from previous mishearings, no substring extraction from
        questions/negation, no correction during task playback or standby.
        """
        value=command_text(text)
        if re.search(r'不|别|吗|么|什么|为何|为什么|或者|还是|如果|假如',value):return None
        prefixes=('给我播放','给我唱一首歌','播放','开始','唱一首','唱','表演一下','表演','来一段','来一首','我想听','我要听')
        for prefix in sorted(prefixes,key=len,reverse=True):
            if value.startswith(prefix):value=value[len(prefix):];break
        if not 4<=len(value)<=12:return None
        candidates={p['id'] for p in self.programs for name in p['aliases']
            if len(normalize(name))==len(value)
            and not normalize(name).startswith(('播放','表演','演','来','作品','节目','歌曲'))
            and sum(a!=b for a,b in zip(value,normalize(name)))==1}
        if len(candidates)>1:return dict(kind='prompt',name='clarify')
        if candidates:return dict(kind='select',program=candidates.pop(),recognition_match='single_character_title')
        return None

    def _command(self,text,active=False,current_program=None):
        value=command_text(text)
        if re.fullmatch(r'(停止|结束|退出)(?:\1)+',value):return dict(kind='interaction',reason=text)
        if re.fullmatch(r'(?:先)?自己玩(?:一?会儿?)?(?:去)?',value):return dict(kind='standby',reason=text)
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
        category=None
        if re.fullmatch(r'(?:给我)?(?:唱{1,4}|播放|放|来)(?:一?首)?歌曲?|播放预设歌曲',value):category='song'
        if re.fullmatch(r'(?:给我)?(?:播放|表演|演|来)(?:一个|一段|个|段)?(?:预设)?剧本(?:表演)?',value):category='script'
        if category:
            choices=[p for p in self.programs if p.get('category', 'song' if p.get('work')=='0001' else None)==category]
            return dict(kind='select',program=choices[0]['id']) if len(choices)==1 else dict(kind='prompt',name='clarify')
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
        if residual and residual not in ('播放','开始','唱','唱一首','来一首','来一段','来段','来首','来个','表演','表演一下','我要听',
            '我想听','我想看','想听','想看','我要看','切换到','切换','换成','换到','给我唱','给我播放','请播放','给我唱一首歌','给我唱一首','唱一首歌'):
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
