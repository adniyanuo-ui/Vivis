"""Narrow recovery of final questions rejected by semantic turn detection."""
import re
from difflib import SequenceMatcher


def complete_question(text):
    if not isinstance(text,str):return False
    compact=re.sub(r'[\W_]+','',text)
    if not 4<=len(compact)<=500:return False
    # This admits an inquiry for Qwen to interpret; it never selects an action.
    return bool(re.search(r'[？?]|吗|了没|过没|有无|是否|什么|怎么|多少|哪|谁',text))


def explicit_expression_request(text):
    """Admit a narrow final request for Qwen interpretation, never raw laughter.

    No negations, quotations, narratives, compound business requests, or motor
    commands. The existing expression tool still validates the actual weights.
    """
    if not isinstance(text,str) or len(text)>80:return False
    compact=re.sub(r'[\s，,。.!！?？、]+','',text)
    return bool(re.fullmatch(
        r'(?:小帅)?(?:请)?(?:你)?(?:给我|帮我)?(?:来个|来一个|做个|做一个|表演一个)?'
        r'(?:(?:哈){2,12}的?大笑|(?:微笑|大笑|笑|哭|悲伤|惊讶)(?:一个|一下))'
        r'(?:吧|呀|啊|嘛)?',compact))


def distinct_from_playback(text,transcripts):
    """Conservative echo rejection, not speaker recognition or intent inference."""
    clean=lambda s:re.sub(r'[\W_]+','',s).casefold()
    if not isinstance(text,str) or len(text)>500:return False
    user=clean(text)
    if not user or not transcripts or any(not isinstance(t,str) or not t.strip() for t in transcripts):return False
    for transcript in transcripts:
        robot=clean(transcript)
        if user in robot:return False
        # Normalize by user length: an echoed sentence in a long answer must
        # not look dissimilar merely because the full answer is longer.
        if len(robot)>8000:return False
        blocks=SequenceMatcher(None,user,robot,autojunk=False).get_matching_blocks()
        if sum(b.size for b in blocks)/len(user)>=.75 or max(b.size for b in blocks)/len(user)>=.6:return False
    return True
