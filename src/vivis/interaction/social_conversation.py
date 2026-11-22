"""Stable observable face cues and public conversation hints; no devices/actions."""
import json
from pathlib import Path
import re

from vivis.paths import ROOT
from vivis.interaction.expression_intent import configuration as expression_configuration
from vivis.interaction.social_expression import SocialExpressionResponse


CONFIG='configs/interaction/social_conversation.json'
STYLES=('neutral','joy','concern','listening','surprise')
PATTERNS=dict(neutral='未观察到稳定的突出动作',joy='双侧嘴角上扬',concern='内眉上提并伴嘴角下压',
    listening='眉部收拢',surprise='抬眉睁眼')
TONE={
    'neutral':'温暖自然，清晰正常语速，不刻意表演情绪。',
    'joy':'轻快、带笑意，有一点活泼；短句有自然停顿，不刻意大笑或突然提高音量。',
    'concern':'温和、关切，先接住对方的话；句间稍留空间，保持正常语速，不拖腔、不强行逗笑。',
    'listening':'认真、耐心、平稳，少抢着给建议，简短回应后给对方说话空间。',
    'surprise':'有兴趣、略带明亮的惊喜感，克制自然，不夸张喊叫。',
}
TONE_POLICY=(' 语气与表情联动：用户本句内容、声音中的表达和明确偏好优先，稳定面部动作仅作辅助。'
    '嘴角上扬也可能是苦笑；若用户抱怨或谈严肃内容，应温和关切或认真倾听，不因看见笑而欢呼。'
    '不要把皱眉、抬眉或微笑说成已知的内心情绪，不说“您看起来不开心/生气了”。'
    '保持当前已选男声音色、语言/方言、正常语速和音量，通过语调、重音和短暂停顿表达分寸。'
    '需要表情时沿用set_expression(source=social)，结合当前话语选择，不每句都调用；明确用户表情口令优先。')
# Public prompts never repeat private business or identifying numbers. Historical
# messages contribute category labels only; current speech is screened separately.
PRIVATE=re.compile(r'\d{4,}|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|https?://|留言|转告|带句话|带个话|取件|快递|验证码|'
    r'手机号|电话|身份证|住址|收入|工资|薪资|财务|婚|恋爱|家庭|家里|病|健康|隐私|秘密|保密')
TOPICS={
    'music':('音乐',('歌曲','唱歌','音乐','听歌')),
    'sport':('轻松运动',('篮球','羽毛球','运动')),
    'expression':('机器人表情',('机器人','表情','仿生')),
    'performance':('节目体验',('剧本','节目','表演')),
    'technology':('一般技术交流',('算法','嵌入式','工程','研发','开发')),
    'demonstration':('产品展示',('销售','市场','产品','接待')),
    'relax':('轻松闲聊',()),
}


def configuration(reference=CONFIG,root=ROOT):
    if reference is None:
        return dict(schema_version='vivis-social-conversation-1.0',tone_enabled=False,topics_enabled=False,
            topic_expression_enabled=False,expression_config='configs/interaction/interaction.json',topic_max_chars=80)
    path=(Path(root)/reference).resolve()
    if not path.is_relative_to(Path(root).resolve()):raise ValueError('Social conversation config must be local')
    c=json.loads(path.read_text())
    if (c.get('schema_version')!='vivis-social-conversation-1.0'
            or any(type(c.get(k)) is not bool for k in ('tone_enabled','topics_enabled','topic_expression_enabled'))
            or not isinstance(c.get('expression_config'),str)
            or type(c.get('topic_max_chars')) is not int or not 30<=c['topic_max_chars']<=100):
        raise ValueError('Invalid social conversation configuration')
    return c


def tone_instructions(style):
    if style not in STYLES:raise ValueError('Unknown social speech style')
    return TONE_POLICY+' 当前辅助语气建议：'+TONE[style]


class SocialConversation:
    def __init__(self,c,max_age_s,root=ROOT):
        self.c=c;self.max_age=max_age_s
        path=(Path(root)/c['expression_config']).resolve()
        if not path.is_relative_to(Path(root).resolve()):raise ValueError('Expression policy must be local')
        self.expression_policy=json.loads(path.read_text())['social_response']['expression_responses']
        self.expressions=expression_configuration(root)
        self.key=None;self.reset()

    def reset(self):
        self.engine=SocialExpressionResponse(self.expression_policy,self.expressions['presets'],
            self.expressions['attack_s'],self.expressions['release_s'],self.max_age)
        self.cue=dict(available=False,response_style='neutral',observation_only=True)
        self.key=None

    def observe(self,packet,now,participant,*,active):
        valid=(active and isinstance(packet,dict) and packet.get('schema_version')=='vivis-face-presence-1.0'
            and type(packet.get('faces')) is int and packet['faces']==1 and type(packet.get('observed_monotonic')) in (int,float)
            and 0<=now-packet['observed_monotonic']<=self.max_age)
        if not valid:self.reset();return self.cue
        key=(participant,packet.get('owner'))
        if self.key!=key:self.reset();self.key=key
        expression=packet.get('expression')
        event=expression.get('patterns') if isinstance(expression,dict) else None
        if isinstance(event,dict) and event.get('timestamp_monotonic')!=packet['observed_monotonic']:event=None
        self.engine.step(event,now)
        available=self.engine.diagnostic.get('valid',False)
        style=self.engine.state if available else 'neutral'
        self.cue=dict(available=available,response_style=style,observation_only=True,
            facial_action=PATTERNS[style])
        return self.cue

    def instructions(self):
        if not self.c['tone_enabled']:return ''
        return (TONE_POLICY+' 当前辅助语气建议仅在本句话语与之相符时参考：'+TONE[self.cue['response_style']]
            +'。如果用户表示疲惫、烦恼、严肃或想安静，忽略欢快建议，先温和接住，不能强行逗笑。')


def public_context(knowledge,person,previous_turns=(),current_turns=(),used_topics=()):
    """Project history/profile into public categories, never copy private records."""
    topic_hints=[]
    profile=knowledge.people.get(person,{}) if person else {}
    texts=[profile.get('role',''),profile.get('professional_summary','')]
    texts += [t.get('text','') for t in previous_turns if t.get('role')=='user']
    for text in texts:
        if not isinstance(text,str):continue
        for key,(_,words) in TOPICS.items():
            if words and any(word in text for word in words) and key not in topic_hints:topic_hints.append(key)
    current=[]
    for turn in list(current_turns)[-6:]:
        text=turn.get('text','')
        if turn.get('role') not in ('user','assistant') or not isinstance(text,str) or PRIVATE.search(text):continue
        if len(text)>180:continue  # Do not truncate a long/private story into a misleading claim.
        current.append(dict(role=turn['role'],text=text))
    allowed=[key for key in TOPICS if key not in used_topics]
    return dict(allowed_topics={key:TOPICS[key][0] for key in allowed},previous_public_topic_hints=topic_hints,
        current_public_turns=current,person_known=bool(profile),
        persona=knowledge.persona,company_background=knowledge.business)


def validate_topic(value,context,max_chars):
    if not isinstance(value,dict) or set(value)!={'topic','text','style'}:raise ValueError('Invalid public topic fields')
    topic,text,style=(value[k] for k in ('topic','text','style'))
    if not isinstance(topic,str) or not isinstance(text,str) or style not in STYLES:raise ValueError('Invalid public topic types')
    if topic=='skip' and text=='' and style=='neutral':return value
    if (topic not in context['allowed_topics'] or not 3<=len(text)<=max_chars
            or text.strip()!=text or '\n' in text or text.count('？')+text.count('?')!=1
            or not text.endswith(('？','?')) or PRIVATE.search(text)):
        raise ValueError('Public invitation must be one bounded, non-private question')
    return value
