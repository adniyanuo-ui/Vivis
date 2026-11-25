"""Presence-entry policy: cloud semantics, bounded greeting evidence only.

This is not an offline command recognizer. The final-text check only recovers
audio/wake permission for direct greetings already transcribed by Qwen,
optionally followed by a bounded contact check such as“有人在吗”.
"""
import json
import re

from vivis.paths import ROOT


def configuration():
    c = json.loads((ROOT/'configs/interaction/presence_entry.json').read_text())
    if (c.get('schema_version') != 'vivis-presence-entry-1.0'
            or type(c.get('enabled')) is not bool or type(c.get('visible_greeting_wake')) is not bool):
        raise ValueError('Invalid presence entry policy')
    for key in ('greetings', 'stop_examples', 'standby_examples'):
        values = c.get(key)
        if (not isinstance(values,list) or not 1 <= len(values) <= 32
                or any(not isinstance(v,str) or not v.strip() or len(v)>40 for v in values)
                or len(set(values)) != len(values)):
            raise ValueError('Invalid presence entry examples: '+key)
    checks=c.get('contact_checks',[])
    if (not isinstance(checks,list) or len(checks)>16
            or any(not isinstance(v,str) or not v.strip() or len(v)>40 for v in checks)
            or len(set(checks))!=len(checks)):
        raise ValueError('Invalid presence entry contact checks')
    return c


def whole_greeting(text, config):
    if (not config['enabled'] or not config['visible_greeting_wake']
            or not isinstance(text,str) or not 1 <= len(text) <= 80):
        return False
    value = re.sub(r'[\s，,。.!！?？、；;]+','',text).casefold()
    words='|'.join(re.escape(v.casefold()) for v in sorted(config['greetings'],key=len,reverse=True))
    # Keep the entire utterance anchored. A contact check is not permission to
    # accept an arbitrary suffix, another addressee, quotation or instruction.
    greeting=r'(?:(?:'+words+r')(?:呀|啊|哟|哦|哈)*){1,4}'
    checks='|'.join(re.escape(v.casefold()) for v in sorted(config.get('contact_checks',[]),key=len,reverse=True))
    suffix=r'(?:(?:'+checks+r')(?:呀|啊|哟|哦|哈)*)?' if checks else ''
    return re.fullmatch(greeting+suffix,value) is not None


def instructions(config):
    if not config['enabled']:
        return (' 待机仅接受配置的明确唤醒词；wake_reason=name，explicit_wake=true。'
                '停止／结束回交互，待机／自己玩回待机。')
    checks=config.get('contact_checks',[])
    contact_hint=('招呼后附带简短询问是否在场（如“你好，'+checks[0]+'？”）仍是直接招呼，同样先wake再有声回应。' if checks else '')
    return (' 交流进退规则：语音意图仍由你理解，不是关键词命中就执行。'
        '待机时，明确呼唤配置名字可用robot_program wake，wake_reason=name且explicit_wake=true。'
        '仅当controller_state.greeting_wake_allowed=true，用户明确面对你打招呼（例如'+
        '、'.join(config['greetings'])+'），也可wake，wake_reason=greeting且explicit_wake=false。'
        '连续、重复及中英组合的直接招呼（如hello，你好）同样先调用wake，不能只生成待机下无声的回答。'
        +contact_hint+
        'greeting_wake_allowed=false时绝不能用greeting调用唤醒；“他刚才说你好”是叙述，不是对你问好，不调用wake。'
        '摄像头持续看到人脸时控制器会主动唤醒，不要求先认出姓名。'
        '仅看到人体但看不到脸时，必须听到对你的直接招呼才能唤醒；看到人体本身不会自动唤醒。'
        '画面无人时只有明确名字呼叫才能唤醒，泛化招呼不能取得画面无人时的说话权限。'
        '人体检测不确认姓名，不授权读取私人资料；引用别人的招呼、对旁人讲话、音箱自播都不是唤醒。'
        '任务中用户表达结束当前表演（例如'+ '、'.join(config['stop_examples'])+
        '），调用stop回交互；“可以了／好了”在普通聊天中可能只是回应，不要无故退出。'
        '用户对你明确表达离开或让你休息（例如'+ '、'.join(config['standby_examples'])+
        '），调用standby回待机；意思相同的自然说法也适用。'
        '不要停止、别结束、不要休息、引述别人和谈论某个词的意思都不执行。'
        '停止和待机不要求人脸、熟人身份或先叫名字；以控制器返回结果为准，不预先声称完成。')
