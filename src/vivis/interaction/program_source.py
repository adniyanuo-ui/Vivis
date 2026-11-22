"""Veto clear negations/quotations; this module never proposes or executes an action."""
import re


TARGETS={
    'stop':r'(?:停止|停下|停一停|停一下|停|结束|退出|打断)',
    'standby':r'(?:待机|休息|休眠|睡觉|自己玩|关闭)',
    'play':r'(?:播放|放|唱|演|开始|换歌|换一首|模仿|跟随)',
}


def veto(text,action):
    if not isinstance(text,str) or not text.strip():return 'missing_final_transcript'
    if len(text)>1000:return 'oversized_final_transcript'
    value=re.sub(r'\s+','',text)
    if re.match(r'^(?:歌词(?:里|中|写|唱)|台词|剧本(?:里|中)|他说|她说|我只是(?:引用|转述)|如果我说)',value):
        return 'quoted_or_hypothetical_control'
    target=TARGETS.get(action)
    # A correction can reverse a request only if it refers to this action.
    # "不要停止，而是继续播放" still vetoes stop. A positive suffix does
    # not authorize anything: the original Qwen proposal remains necessary.
    suffix=re.split(r'不对[，,、]?|改口[，,、]?|还是[，,、]?|而是[，,、]?',value)[-1]
    if target and re.search(target,suffix):value=suffix
    if target and re.search(r'(?:不要|不用|不必|不许|不准|不能|不想|不希望|别|请勿|无需|不是让你|没让你)'
            r'(?:你|您|再|先|马上|立刻|给我|帮我|把歌|把歌曲|把节目){0,3}'+target,value):
        return 'negated_control'
    return None
