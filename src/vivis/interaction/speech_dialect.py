"""Visit-scoped Chinese speech preference. No ASR rules or device access."""
from functools import lru_cache
import json

from vivis.paths import ROOT


@lru_cache(maxsize=1)
def configuration():
    config=json.loads((ROOT/'configs/perception/speech_dialects.json').read_text())
    names=config.get('dialects')
    if (config.get('schema_version')!='vivis-speech-dialects-1.0'
            or not isinstance(names,list) or not names
            or any(not isinstance(n,str) or not n.strip() or len(n)>20 for n in names)
            or len(set(names))!=len(names) or config.get('default') not in names
            or config.get('dialect_voice')!='xunanchuan_v3.1'
            or not isinstance(config.get('documented_dialects'),list)
            or any(n not in names for n in config['documented_dialects'])):
        raise ValueError('Invalid speech dialect configuration')
    return config


class SpeechDialect:
    def __init__(self):
        self.c=configuration();self.name=self.c['default']

    def requested(self,args):
        if (not isinstance(args,dict) or set(args)!={'dialect'}
                or not isinstance(args['dialect'],str) or args['dialect'] not in self.c['dialects']):
            raise ValueError('Choose one supported canonical dialect; only dialect is allowed')
        return args['dialect']

    def reset(self):self.name=self.c['default']

    def voice(self,default_voice):return default_voice if self.name==self.c['default'] else self.c['dialect_voice']

    def snapshot(self):return dict(dialect=self.name,language='zh',scope='current_visit',
        capability='documented' if self.name==self.c['default'] or self.name in self.c['documented_dialects'] else 'experimental')

    def instructions(self):
        return (' 当前有效的语音偏好（优先于人设、历史记录中的默认普通话/北京口音要求，只覆盖说话语言风格）：'
            +json.dumps(self.snapshot(),ensure_ascii=False)
            +'。从下一次回复起使用该方言的自然词汇、声调、韵母和节奏，不是用普通话读方言文字；正常语速和人物身份不变。'
            '普通话保留原默认男声；用户已同意方言模式使用许南川男声，由本地控制器重连并确认音色，不自行声称已换好。'
            '只有用户明确要求切换/恢复时才调用set_speech_dialect；用户偶尔说方言不等于要求切换。'
            '粤语对应广东话，沪语对应上海话；恢复普通话选择普通话。不要把陕西话与山西话混淆。'
            '切换必须先调用工具，accepted=false时不能声称已切换。工具成功只证明指令已更新，不证明口音准确。'
            '未经听感验收不声称正宗、地道；不能把普通话朗读方言词说成成功切换，也不以北京身份掩饰音色能力限制。'
            'capability=experimental表示官方音色表没有明确保证这种方言，需试听，不承诺地道。'
            '不改写预录歌曲或剧本，不因切换方言清除同一访客的近期聊天。')
