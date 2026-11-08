"""Optional text-only semantic routing. No audio, hardware or state mutation."""
import json
import math
import time
import urllib.request
from urllib.parse import urlsplit

from vivis.runtime.best_effort import BestEffortJob


def response_schema(catalog):
    return dict(type='object',additionalProperties=False,required=['action','program','reply'],properties=dict(
        action=dict(type='string',enum=['select','menu','identity','chat','clarify','none']),
        program=dict(type='string',enum=['']+[p['id'] for p in catalog['programs']]),
        reply=dict(type='string',maxLength=100)))


class SemanticRouter:
    def __init__(self, config, catalog, identity=None):
        if config.get('schema_version')!='vivis-semantic-1.0':raise ValueError('Unsupported semantic configuration')
        self.config=config;self.catalog=catalog;self.ids={p['id'] for p in catalog['programs']};self.identity=identity or {}
        timeout=config.get('timeout_s',4.)
        if type(timeout) not in (int,float) or not math.isfinite(timeout) or not .1<=timeout<=15:
            raise ValueError('Invalid semantic request timeout')
        endpoint=urlsplit(config.get('endpoint',''))
        if (endpoint.scheme!='http' or endpoint.hostname!='127.0.0.1' or endpoint.username is not None
                or endpoint.password is not None or endpoint.port is None or endpoint.query or endpoint.fragment):
            raise ValueError('The initial semantic provider must be a loopback HTTP service')

    def payload(self, text):
        menu=[dict(id=p['id'],title=p['title'],aliases=p['aliases']) for p in self.catalog['programs']]
        system=(self.config['persona']+'\n本体身份：'+json.dumps(self.identity,ensure_ascii=False)+
            '\n你只负责理解用户当前这一句话，输出符合schema的JSON。'
            '可选节目：'+json.dumps(menu,ensure_ascii=False)+'。'
            '明确请求点播/模仿才select；不确定或同时点多个节目用clarify。'
            '不要把否定、假设、引用他人话语、询问某节目是什么当作点播。'
            '先区分这句话的言语行为：提到节目不等于要求现在执行；能不能为我表演属于请求。'
            '询问功能、能力、本事、能帮什么忙都用menu；询问名字、年龄、来历用identity。'
            '解释词义、假设讨论和一般聊天用chat，无意义语气词用none且reply为空。'
            'program只有select时填写节目id，其他时候必须为空。reply用简短自然普通话，最多两句话。'
            'select的reply只用一句简短的准备回应，不评价作品、不承诺顺序播放多个节目。'
            '你不能控制硬件、创建新功能或声称已执行动作；不得承诺不存在的联网查询能力。'
            '停止/退出表示返回交互，待机表示关摄像头等待唤醒；这些由本地控制器处理，你无权覆盖。/no_think')
        messages=[dict(role='system',content=system)]
        if len(self.catalog['programs'])>=2:
            a,b=self.catalog['programs'][:2]
            examples=[
                ('假设我让你表演'+a['title']+'，你会怎么回应？','chat','','我会先准备，再开始表演。'),
                ('别播放'+a['title']+'，只是问问有什么节目','menu','','我可以介绍目前的节目。'),
                ('有人喊着要看'+b['title']+'，我是在转述。','chat','','明白，你是在转述。'),
                ('先给我'+a['title']+'，再来'+b['title'],'clarify','','想先看哪一个节目？'),
                ('让我看看你都能做啥，有啥绝活儿','menu','','我可以介绍一下节目单。'),
                ('能不能为我表演'+a['title']+'？','select',a['id'],'好，我来准备。')]
            for utterance,action,program,reply in examples:
                messages.extend([dict(role='user',content=utterance),
                    dict(role='assistant',content=json.dumps(dict(action=action,program=program,reply=reply),ensure_ascii=False))])
        messages.append(dict(role='user',content=text))
        return dict(model=self.config['model'],messages=messages,
            temperature=.1,max_tokens=180,stream=False,chat_template_kwargs=dict(enable_thinking=False),
            response_format=dict(type='json_schema',json_schema=dict(name='program_intent',strict=True,schema=response_schema(self.catalog))))

    def validate(self, result):
        if (not isinstance(result,dict) or set(result)!={'action','program','reply'}
                or not isinstance(result['reply'],str) or len(result['reply'])>100
                or result['action'] not in ('select','menu','identity','chat','clarify','none')):
            raise ValueError('Invalid semantic response')
        action=result['action'];program=result['program']
        if action=='select':
            if not isinstance(program,str) or program not in self.ids:raise ValueError('Unregistered semantic program')
            intent=dict(kind='select',program=program)
        else:
            if program!='':raise ValueError('Non-selection cannot carry a program')
            intent=dict(kind='prompt',name=action) if action in ('menu','identity','clarify') else dict(kind='ignore')
        return dict(intent=intent,reply=result['reply'].strip())

    def resolve(self, text):
        if not isinstance(text,str) or not 1<=len(text)<=256:raise ValueError('Invalid semantic input text')
        started=time.monotonic()
        request=urllib.request.Request(self.config['endpoint'],data=json.dumps(self.payload(text)).encode(),
                                       headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=self.config['timeout_s']) as response:
            raw=response.read(65537)
        if len(raw)>65536:raise ValueError('Oversize semantic response')
        envelope=json.loads(raw);content=envelope['choices'][0]['message']['content']
        result=self.validate(json.loads(content));result['seconds']=time.monotonic()-started
        result['usage']=envelope.get('usage',{})
        return result


class SemanticWorker:
    def __init__(self, router, emit, event_kind='semantic_result'):
        self.router=router;self.emit=emit;self.job=BestEffortJob(event_kind);self.epoch=0;self.event_kind=event_kind

    def invalidate(self):self.epoch+=1

    def offer(self, event, mode):
        if mode!='interaction':return False
        epoch=self.epoch
        expires=time.monotonic()+self.router.config['timeout_s']+.5
        def run():
            result=dict(kind=self.event_kind,epoch=epoch,utterance_id=event['utterance_id'],text=event['text'],mode=mode,
                        expires_monotonic=expires)
            try:result.update(self.router.resolve(event['text']))
            except Exception as exc:result['error']=str(exc)
            if not self.job.closed.is_set():self.emit(result)
        return self.job.offer(run)

    def accepts(self, event, mode):
        return (not self.job.closed.is_set() and event['epoch']==self.epoch and mode=='interaction' and event['mode']==mode
                and time.monotonic()<=event['expires_monotonic'])

    def close(self):
        self.invalidate();self.job.close()


def main(argv=None):
    import argparse
    from pathlib import Path
    import subprocess
    from urllib.parse import urlsplit
    from vivis.paths import ROOT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/interaction/semantic.json')
    parser.add_argument('--robot',type=Path,default=ROOT/'configs/robots/xiaoshuai/robot.json')
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--serve',action='store_true');group.add_argument('--text');group.add_argument('--check',action='store_true')
    args=parser.parse_args(argv);config=json.loads(args.config.read_text())
    catalog=json.loads((ROOT/'configs/interaction/programs.json').read_text())
    robot=json.loads(args.robot.read_text());identity=robot.get('identity',dict(display_name=robot['name']))
    router=SemanticRouter(config,catalog,identity)
    runtime=config['runtime'];folder=ROOT/runtime['directory'];sources=json.loads((ROOT/runtime['source_manifest']).read_text())
    if args.check:
        print(json.dumps(dict(config_valid=True,model_present=(folder/sources['model']['file']).is_file(),
                             server_present=(folder/'build/bin/llama-server').is_file(),hardware_opened=False)));return 0
    if args.text:
        print(json.dumps(router.resolve(args.text),ensure_ascii=False,indent=2));return 0
    if not (folder/sources['model']['file']).is_file() or not (folder/'build/bin/llama-server').is_file():
        parser.error('请先执行 python3 tools/setup_semantic.py 准备本地模型和引擎')
    endpoint=urlsplit(config['endpoint'])
    command=[str(folder/'build/bin/llama-server'),'-m',str(folder/sources['model']['file']),
             '--alias',config['model'],'--host','127.0.0.1','--port',str(endpoint.port),
             '-c',str(runtime['context_tokens']),'-ngl',str(runtime['gpu_layers']),'-t',str(runtime['cpu_threads']),
             '--parallel','1','--jinja','--reasoning','off','--reasoning-budget','0','--no-webui']
    try:return subprocess.call(command,cwd=ROOT)
    except KeyboardInterrupt:return 130


if __name__=='__main__':raise SystemExit(main())
