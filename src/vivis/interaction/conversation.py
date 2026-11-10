"""Online conversation returns text only; no task tools or actuator access."""
import json
import math
import os
import time
import urllib.request
from urllib.parse import urlsplit


class ConversationRouter:
    def __init__(self, config, catalog, identity, knowledge=None):
        if config.get('schema_version')!='vivis-cloud-chat-1.0':raise ValueError('Unsupported cloud chat config')
        self.config=config;self.catalog=catalog;self.identity=identity;self.history=[];self.knowledge=knowledge
        self.provider=config['provider'];self.service=config['providers'][self.provider]
        endpoint=urlsplit(self.service['endpoint']);host=endpoint.hostname or ''
        if (endpoint.scheme!='https' or endpoint.username or endpoint.password
                or (self.provider=='deepseek' and host!='api.deepseek.com')
                or (self.provider=='qwen' and host not in ('dashscope.aliyuncs.com','dashscope-intl.aliyuncs.com','dashscope-us.aliyuncs.com')
                    and not host.endswith('.maas.aliyuncs.com'))
                or self.provider not in ('deepseek','qwen')):
            raise ValueError('Use the configured provider official HTTPS endpoint')
        timeout=config['timeout_s']
        if type(timeout) not in (int,float) or not math.isfinite(timeout) or not .1<=timeout<=30:raise ValueError('Invalid cloud timeout')
        if type(config['history_turns']) is not int or not 0<=config['history_turns']<=8:raise ValueError('Invalid conversation history bound')
        if type(config['max_reply_chars']) is not int or not 1<=config['max_reply_chars']<=1000:raise ValueError('Invalid reply bound')

    def payload(self, text):
        titles='、'.join(p['title'] for p in self.catalog['programs'])
        system=(self.config['persona']+'\n本体身份：'+json.dumps(self.identity,ensure_ascii=False)+
            '。目前节目：'+titles+'。角色扮演、前台接待、导游尚未实现。'
            '你只负责聊天、情绪回应、解释和讲笑话，没有控制机器人的工具。'
            '不能声称已开始/停止节目、移动身体或修改设置；若用户想执行节目，请让他直接说播放节目名。'
            '停止/退出回交互，待机/休息回关闭摄像头的待机，由本地控制器处理。'
            '不虚构联网搜索结果或记忆。回答保持简短，最多'+str(self.config['max_reply_chars'])+'字。')
        if self.knowledge is not None:
            system+=self.knowledge.base_instructions()+self.knowledge.person_instructions()
        body=dict(model=self.service['model'],messages=[dict(role='system',content=system),*list(self.history),dict(role='user',content=text)],
                  max_tokens=256,temperature=.7,stream=False)
        if self.provider=='deepseek':body['thinking']=dict(type='disabled')
        else:body['enable_thinking']=False
        return body

    def resolve(self, text):
        if not isinstance(text,str) or not 1<=len(text)<=256:raise ValueError('Invalid conversation input')
        key=os.environ.get(self.service['api_key_env'])
        if not key:raise RuntimeError('请在本机设置环境变量 '+self.service['api_key_env']+'；不要把密钥发到聊天或写入仓库')
        started=time.monotonic()
        request=urllib.request.Request(self.service['endpoint'],data=json.dumps(self.payload(text)).encode(),
                    headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
        with urllib.request.urlopen(request,timeout=self.config['timeout_s']) as response:raw=response.read(65537)
        if len(raw)>65536:raise ValueError('Oversize conversation response')
        envelope=json.loads(raw);reply=envelope['choices'][0]['message'].get('content')
        if not isinstance(reply,str) or not reply.strip():raise ValueError('No conversation text returned')
        reply=reply.strip()[:self.config['max_reply_chars']]
        # Even JSON-looking text or a model tool proposal is never interpreted as an action.
        return dict(intent=dict(kind='ignore'),reply=reply,seconds=time.monotonic()-started,usage=envelope.get('usage',{}))

    def commit(self, text, reply):
        if not self.config['history_turns']:return
        self.history.extend([dict(role='user',content=text),dict(role='assistant',content=reply)])
        self.history=self.history[-2*self.config['history_turns']:]

    def clear(self):self.history=[]


def main(argv=None):
    import argparse
    from pathlib import Path
    from vivis.paths import ROOT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/interaction/conversation.json')
    parser.add_argument('--robot',type=Path,default=ROOT/'configs/robots/xiaoshuai/robot.json')
    group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--text');group.add_argument('--check',action='store_true')
    args=parser.parse_args(argv);config=json.loads(args.config.read_text())
    catalog=json.loads((ROOT/'configs/interaction/programs.json').read_text());robot=json.loads(args.robot.read_text())
    from vivis.interaction.knowledge import Knowledge
    router=ConversationRouter(config,catalog,robot.get('identity',dict(display_name=robot['name'])),Knowledge(args.robot))
    if args.check:
        print(json.dumps(dict(config_valid=True,provider=router.provider,model=router.service['model'],
                             enabled=config['enabled'],hardware_opened=False,network_called=False)));return 0
    print(json.dumps(router.resolve(args.text),ensure_ascii=False,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
