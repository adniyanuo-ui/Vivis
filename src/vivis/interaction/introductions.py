"""Durable, source-bound introductions; no camera/model or device ownership."""
import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid


def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as stream:
            json.dump(data,stream,ensure_ascii=False,allow_nan=False);stream.flush();os.fsync(stream.fileno())
        temporary.replace(path)
    finally:temporary.unlink(missing_ok=True)


def text(value,limit):
    if not isinstance(value,str) or len(value)>limit or any(ord(c)<32 for c in value):
        raise ValueError('介绍字段格式或长度不合适')
    return value.strip()


class Introductions:
    def __init__(self,root):
        self.root=Path(root);self.directory=self.root/'private/introductions'

    def request(self,information,*,guest,introducer,source,session,now=None):
        now=time.monotonic() if now is None else now
        if not isinstance(information,dict):raise ValueError('介绍资料需要提供姓名、称呼和背景字段')
        if (not isinstance(guest,dict) or not re.fullmatch(r'[a-f0-9]{32}',guest.get('token',''))
                or not re.fullmatch(r'[a-f0-9]{32}',guest.get('owner',''))
                or not 0<=now-guest.get('stamp',float('-inf'))<=1.):
            raise ValueError('眼前访客已变化或画面证据过期，请重新介绍眼前这位')
        if guest.get('person_id'):raise ValueError('眼前访客已有身份，请更新已有档案，不能重复登记')
        if not re.fullmatch(r'[a-z][a-z0-9_-]{1,39}',introducer or ''):raise ValueError('介绍人本轮声纹尚未确认')
        if not isinstance(source,dict) or not source.get('turn_id') or not source.get('text'):
            raise ValueError('介绍缺少本轮原话来源')
        if set(information)-{'display_name','preferred_address','background'}:raise ValueError('未知介绍字段')
        name=text(information.get('display_name',''),40);address=text(information.get('preferred_address',''),40)
        background=text(information.get('background',''),500)
        if not name and not address:return dict(accepted=False,status='needs_address',notice='这位老师怎么称呼？')
        # The name/address must come from the actual introduction, not a model
        # completion of a title. Background remains attributed, unverified prose.
        original=source['text']
        compact=lambda s:re.sub(r'[\s，。！？、,.!?：:；;]','',s)
        if name and compact(name) not in compact(original):raise ValueError('姓名必须来自介绍原话，请不要补全姓名')
        if address and compact(address) not in compact(original):
            # “杨明扬教授” explicitly supports the ordinary address “杨教授”.
            # A title absent from the source must never be inferred from a job.
            title=address[1:] if name and address.startswith(name[0]) else None
            if title not in ('教授','老师','先生','女士','经理','总') or compact(name+title) not in compact(original):
                raise ValueError('称呼必须来自介绍原话或原话明确给出的姓名和头衔')
        key=hashlib.sha256((session+'|'+guest['owner']+'|'+guest['token']).encode()).hexdigest()
        path=self.directory/'requests'/f'{key}.json'
        if path.exists():
            previous=json.loads(path.read_text())
            if previous['information']!=dict(display_name=name,preferred_address=address,background=background):
                raise ValueError('这位访客已经登记了介绍；资料更正需明确提出，不能重试时覆盖')
            return dict(accepted=previous['status']!='failed',status=previous['status'],person_id=previous['person_id'],
                preferred_address=address or name,notice=previous.get('error','该介绍已登记，不重复创建'))
        # A matching label is not proof of the same person. Do not silently merge.
        for profile in (self.root/'private/people').glob('*/profile.json'):
            data=json.loads(profile.read_text())
            if any(v and v in (data.get('display_name'),data.get('preferred_address')) for v in (name,address)):
                raise ValueError('已有同名称呼的档案，请补充一个能区分的称呼，不能自动合并')
        pid='v_'+uuid.uuid4().hex
        request=dict(schema_version='vivis-introduction-1.0',id=key,person_id=pid,status='pending_face',
            guest=guest,introducer=introducer,session=session,created_monotonic=now,expires_monotonic=now+120,
            created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            information=dict(display_name=name,preferred_address=address,background=background),
            source=dict(turn_id=source['turn_id'],text=original[:4000]),background_verified=False)
        atomic(path,request)
        return dict(accepted=True,status='pending_face',person_id=pid,preferred_address=address or name,
            notice='介绍已收到，可以使用称呼欢迎；初始人脸模板后台登记中，尚不能宣称认脸成功。')

    def statuses(self):
        paths=sorted((self.directory/'requests').glob('*.json'),key=lambda p:p.stat().st_mtime_ns)[-100:]
        return [json.loads(p.read_text()) for p in paths]

    def maintain(self,now=None):
        now=time.monotonic() if now is None else now
        manifest=self.directory/'face_anchors.json'
        anchors=json.loads(manifest.read_text())['people'] if manifest.exists() else {}
        rows=self.statuses()
        for row in rows:
            if row['status']!='pending_face':continue
            if anchors.get(row['person_id'],{}).get('source')==row['id']:
                row.update(status='ready')
            elif not 0<=now-row['created_monotonic']<=120:
                row.update(status='failed',error='未取得当前访客的合格单人照片，请正对摄像头后重新介绍')
            else:continue
            atomic(self.directory/'requests'/f"{row['id']}.json",row)
        return rows
