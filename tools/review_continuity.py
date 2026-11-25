#!/usr/bin/env python3
"""Export a PRIVATE, read-only photo/audio review page. No devices or server."""
import argparse
from datetime import datetime
import html
import json
import os
from pathlib import Path
import shlex
import sqlite3
import sys
import tempfile
from urllib.parse import quote

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vivis.paths import ROOT
from vivis.interaction.memory import person_key


def esc(value):return html.escape(str(value),quote=True)


def status(value):
    return {'complete':'已完成','interrupted':'已被打断','open':'进行中','playing':'播放中',
        'next_user_turn':'已进入下一轮','connection_ended':'连接已结束','process_interrupted':'进程中断',
        'text_only':'只有文字','aligned':'边界已关联','recording':'录音中','gapped':'有缺帧',
        'recording_queue_gap':'录音队列丢帧','recovered_invalid_turn_partial':'恢复的片段（非完整）',
        'server_offset_unavailable':'缺少输入时间边界','end_offset_unavailable':'缺少结束边界',
        'missing_preroll':'开头录音不完整','unanswered':'未取得回答','failed':'失败','cancelled':'已取消'}.get(value,value)


def local_time(value):
    try:return datetime.fromisoformat(value).astimezone().strftime('%Y-%m-%d %H:%M:%S')
    except (TypeError,ValueError):return value


def export(root,output,rounds=32,person=None):
    root=Path(root).resolve();private=root/'private';output=Path(output).resolve()
    if not output.is_relative_to(private):raise ValueError('审核页必须保存在本项目private目录下')
    if output.suffix.lower()!='.html':raise ValueError('审核页文件名必须以.html结尾，不能覆盖档案或录音')
    if not 1<=rounds<=500:raise ValueError('每人显示1..500轮')
    if person:person_key(person)
    c=json.loads((root/'configs/interaction/continuity.json').read_text())
    face=json.loads((root/'configs/perception/familiar_faces.json').read_text())
    archive=(root/c['directory']).resolve();people=(root/face['people_directory']).resolve()
    if not archive.is_relative_to(private) or not people.is_relative_to(private):raise ValueError('私有数据路径越界')
    speaker_config=json.loads((root/c['speaker_config']).read_text())
    speaker_root=(root/speaker_config['directory']).resolve()
    if not speaker_root.is_relative_to(private):raise ValueError('私有声纹路径越界')
    speakers={};sp=speaker_root/'gallery.json'
    if sp.exists():speakers=json.loads(sp.read_text()).get('people',{})
    db=None;path=archive/'rounds.sqlite3'
    if path.exists():db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    def href(path):
        path=Path(path).resolve()
        if not path.is_relative_to(private) or not path.is_file():return None
        return esc(quote(os.path.relpath(path,output.parent),safe='/'))
    def audio(path):
        link=href(path)
        return '<audio controls preload="none" src="'+link+'"></audio>' if link else '<span class="muted">录音不存在／未保存</span>'
    parts=['''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>聊天档案与身份样本审核</title><style>
body{font:16px/1.6 system-ui,sans-serif;max-width:1280px;margin:24px auto;padding:0 20px;color:#202a37;background:#f6f7fa}
section,header{background:white;padding:20px;margin:18px 0;border-radius:12px}h1,h2,h3{line-height:1.35}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(205px,1fr));gap:14px}figure{margin:0;padding:10px;background:#f2f5f8;border-radius:8px}
img{width:100%;height:180px;object-fit:contain;background:#e6eaf0}.muted{color:#667085;font-size:14px}.badge{display:inline-block;padding:2px 8px;border-radius:12px;background:#daeedd;margin:4px 0}
.candidate{background:#f5ebcf}.notice{background:#fff2cc;padding:12px;border-radius:8px}code{overflow-wrap:anywhere;font-size:13px}audio{width:100%;max-width:500px}
article{border-top:1px solid #ddd;padding:14px 0}summary{cursor:pointer}nav a{margin-right:18px}td,th{padding:6px 16px;border-bottom:1px solid #ddd;text-align:left}
</style><header><h1>聊天档案与身份样本审核</h1>
<p><a href="http://127.0.0.1:8766/">打开实时样本管理：点击即可停用、激活、锁定或删除照片与声纹</a>（需本机审核服务运行）。</p>
<p>本文件是只读静态快照；不修改照片、声纹、聊天档案或留言状态。点击原图可放大，录音按需加载。下方命令仍可复制到终端执行。</p>
<div class="notice">听录音建议用耳机或另一台电脑。若使用Orin音箱，请先停止交互服务：<code>sudo systemctl stop vivis-p1.service</code>，避免把录音当成新话语；听完后用<code>sudo systemctl start vivis-p1.service</code>恢复。
录音可能含机器人回声，不能把每一条自动当作真人声纹参考。正式交互按图声证据核验身份；普通录音不直接成为已确认身份。</div>''']
    parts.append('<p class="muted">生成时间：'+esc(datetime.now().astimezone().isoformat(timespec='seconds'))+'；内容不会自动刷新，重新运行导出命令可更新。</p></header>')
    roster={}
    for f in sorted(people.glob('*/profile.json')):
        profile=json.loads(f.read_text());roster[f.parent.name]=profile.get('display_name') or f.parent.name
    if db:
        for row in db.execute('SELECT DISTINCT person FROM rounds'):roster.setdefault(row[0],row[0])
    if person:roster={k:v for k,v in roster.items() if k==person}
    parts.append('<nav>'+''.join('<a href="#'+esc(pid)+'">'+esc(name)+'</a>' for pid,name in roster.items())+'</nav>')
    try:
        for pid,name in roster.items():
            person_key(pid);folder=people/pid;pool_path=folder/'photo_pool.json';pool=json.loads(pool_path.read_text()) if pool_path.exists() else dict(rows=[],active=[])
            records=[dict(r) for r in db.execute('SELECT * FROM rounds WHERE person=? ORDER BY created DESC LIMIT ?',(pid,rounds))] if db else []
            total=db.execute('SELECT count(*) FROM rounds WHERE person=?',(pid,)).fetchone()[0] if db else 0
            parts.append('<section id="'+esc(pid)+'"><h2>'+esc(name)+' <small class="muted">'+esc(pid)+'</small></h2><p>照片 '+str(len(pool['rows']))+' 张；活动 '+str(len(pool['active']))+' 张；候选 '+str(len(pool['rows'])-len(pool['active']))+' 张。聊天 '+str(total)+' 轮；声纹参考 '+str(len(speakers.get(pid,[])))+' 段。</p>')
            collection=pool.get('collection') or {}
            progress='已满32张，自动采集停止；仅人工删除后补空缺。' if len(pool['rows'])>=32 else '仅补空缺，无每日限制；不自动替换现有照片。'
            if collection.get('last_batch_day'):progress+=' 最近采集：'+str(collection['last_batch_day'])+'。'
            parts.append('<p class="muted">'+esc(progress)+'</p>')
            for active,label in ((True,'活动照片'),(False,'候选照片')):
                selected=[r for r in pool['rows'] if (r['hash'] in pool['active'])==active]
                if not selected:continue
                parts.append('<details open><summary>'+label+'（'+str(len(selected))+'）</summary><div class="grid">')
                for row in selected:
                    link=href(folder/row['path']);tag=('基础照 · 已锁定' if row['origin']=='manual' else '人工激活 · 已锁定') if row['locked'] else '已停用 · 不自动参与识别' if row.get('excluded') else '自动采集' if row['origin']=='automatic' else '人工提供'
                    view='<a href="'+link+'" target="_blank"><img loading="lazy" src="'+link+'"></a>' if link else '<p>原图不可用</p>'
                    controls=''
                    if not row['locked'] or row['origin']=='automatic':
                        operation='--deactivate' if active else '--activate'
                        command=shlex.join([str(root/'.venv/bin/python'),str(root/'tools/manage_photo_pool.py'),'--person',pid,operation,row['hash']])
                        label='停用指令' if active else '激活指令（活动未满16张时）'
                        controls='<details><summary>'+label+'</summary><code>'+esc(command)+'</code></details>'
                    if active and not row['locked']:
                        command=shlex.join([str(root/'.venv/bin/python'),str(root/'tools/manage_photo_pool.py'),'--person',pid,'--lock',row['hash']])
                        controls+='<details><summary>人工锁定指令</summary><code>'+esc(command)+'</code></details>'
                    if row['origin']=='automatic':
                        command=shlex.join([str(root/'.venv/bin/python'),str(root/'tools/manage_photo_pool.py'),'--person',pid,'--delete',row['hash']])
                        controls+='<details><summary>永久删除指令（清空一个采集名额）</summary><code>'+esc(command)+'</code></details>'
                    parts.append('<figure>'+view+'<figcaption><span class="badge'+(' candidate' if not active else '')+'">'+tag+'</span><br><code>'+esc(row['path'])+'</code><details><summary>完整hash</summary><code>'+esc(row['hash'])+'</code></details>'+controls+'</figcaption></figure>')
                parts.append('</div></details>')
            refs=speakers.get(pid,[])
            parts.append('<details><summary>声纹参考录音（'+str(len(refs))+'段）</summary>')
            maximum=speaker_config['maximum_samples']
            parts.append('<p>'+('已满'+str(maximum)+'段，停止新增；人工删除后只补空缺。' if len(refs)>=maximum else '还缺'+str(maximum-len(refs))+'段合格样本，不限每日批次。')+' 样本操作不直接修改当前业务身份。</p>')
            for r in refs:
                parts.append('<article><code>'+esc(r['turn_id'])+'</code><br>'+audio(speaker_root/pid/(r['turn_id']+'.wav'))+'</article>')
            if not refs:parts.append('<p>尚无合格声纹参考；普通聊天录音与声纹参考不是同一份清单。</p>')
            parts.append('</details><h3>近期音文记录（最近'+str(len(records))+'轮）</h3>')
            for r in records:
                parts.append('<article><p class="muted">'+esc(local_time(r['created']))+' · '+esc(status(r['outcome']))+' · '+esc(status(r['alignment']))+'</p><details><summary>关联详情</summary><code>话轮 '+esc(r['id'])+'<br>会话 '+esc(r['session'])+'</code></details><p><strong>用户：</strong>'+esc(r['user_text'] or '尚无定稿文字')+'</p>')
                if r['audio']:parts.append(audio(archive/r['audio']))
                for reply in db.execute('SELECT text,audio,outcome FROM replies WHERE round_id=?',(r['id'],)):
                    parts.append('<p><strong>小帅：</strong>'+esc(reply['text'] or '未取得完整回答文字')+' <span class="muted">（'+esc(status(reply['outcome']))+'）</span></p>')
                    if reply['audio']:parts.append(audio(archive/reply['audio']))
                parts.append('</article>')
            parts.append('</section>')
    finally:
        if db:db.close()
    parts.append('</html>');output.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    fd,temporary=tempfile.mkstemp(prefix='.review-',suffix='.html',dir=output.parent)
    try:
        with os.fdopen(fd,'w') as f:f.write('\n'.join(parts))
        os.replace(temporary,output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--person');p.add_argument('--rounds',type=int,default=32)
    p.add_argument('--output',type=Path,default=ROOT/'private/review/index.html');a=p.parse_args()
    print(export(ROOT,a.output,a.rounds,a.person))


if __name__=='__main__':main()
