"""Owner-authorized pickup-code messages for self-declared visiting couriers.

No face identity is required to receive these explicitly released codes. This
store never returns personal history or other registered-person messages.
"""
import json
import re
import uuid

from vivis.interaction.memory import person_key
from vivis.interaction.robot_tasks import now_utc


def code_digits(text):
    text=text.translate(str.maketrans('零〇一二两三四五六七八九幺','0012234567891'))
    return re.findall(r'(?<!\d)[0-9]{4,8}(?!\d)',text)


def public_receipt(owner,code):
    return owner+'这件退件的取件码是'+ '、'.join(code)+'。您收到取件码了吗？'


def pickup_request(text):
    return isinstance(text,str) and bool(re.search(r'(?:取|拿|收).{0,20}(?:快递|退件|包裹|件)|(?:取件|退件|退货)码',text))


class CourierHandoffs:
    def __init__(self,memory):self.memory=memory;self.ready=False

    def ensure_schema(self):
        if self.ready:return
        with self.memory.db() as db:
            db.executescript('''CREATE TABLE IF NOT EXISTS courier_handoffs(
                id TEXT PRIMARY KEY,owner TEXT NOT NULL UNIQUE,owner_label TEXT NOT NULL,
                aliases TEXT NOT NULL,code TEXT NOT NULL,status TEXT NOT NULL,
                version INTEGER NOT NULL,source_ids TEXT NOT NULL,created TEXT NOT NULL,updated TEXT NOT NULL);
                CREATE UNIQUE INDEX IF NOT EXISTS courier_active_owner ON courier_handoffs(owner)
                    WHERE status IN ('pending','announced');
                CREATE TABLE IF NOT EXISTS courier_handoff_events(
                id INTEGER PRIMARY KEY,handoff_id TEXT NOT NULL,event TEXT NOT NULL,
                actor TEXT NOT NULL,details TEXT NOT NULL,created TEXT NOT NULL);''')
        self.ready=True

    def event(self,db,hid,event,actor,details):
        db.execute('INSERT INTO courier_handoff_events(handoff_id,event,actor,details,created) VALUES(?,?,?,?,?)',
            (hid,event,actor,json.dumps(details,ensure_ascii=False),now_utc()))

    def register(self,owner,label,aliases,code,source_ids,*,source_reason):
        self.ensure_schema();person_key(owner)
        if not isinstance(code,str) or not re.fullmatch(r'[0-9]{4,8}',code):raise ValueError('取件码需要4～8位数字，请本人明确说明。')
        if not isinstance(label,str) or not 1<=len(label)<=40:raise ValueError('Invalid owner label')
        if (not isinstance(aliases,list) or not aliases or len(aliases)>12
                or any(not isinstance(a,str) or not 1<=len(a)<=40 for a in aliases)):raise ValueError('Invalid owner aliases')
        if not source_ids or len(source_ids)>12 or any(type(mid) is not int or mid<1 for mid in source_ids):raise ValueError('Owner source messages required')
        with self.memory.db() as db:
            db.execute('BEGIN IMMEDIATE')
            rows=[db.execute("SELECT person,role,text,created FROM messages WHERE id=?",(mid,)).fetchone() for mid in source_ids]
            if any(not r or r['person']!=owner or r['role']!='user' for r in rows):raise ValueError('取件码只能来自委托人本人的明确交代。')
            evidence=[r for r in rows if '码' in r['text'] and code_digits(r['text'])]
            if not evidence or code_digits(max(evidence,key=lambda r:r['created'])['text'])[-1]!=code:
                raise ValueError('取件码与本人最新交代不一致，请本人重新确认。')
            previous=db.execute('SELECT * FROM courier_handoffs WHERE owner=?',(owner,)).fetchone()
            if previous and previous['code']==code and json.loads(previous['source_ids'])==source_ids:
                return dict(previous)
            hid=previous['id'] if previous else uuid.uuid4().hex
            if previous:aliases=list(dict.fromkeys(aliases+json.loads(previous['aliases'])))[:12]
            version=(previous['version']+1) if previous else 1;stamp=now_utc()
            if previous:
                db.execute("UPDATE courier_handoffs SET owner_label=?,aliases=?,code=?,status='pending',version=?,source_ids=?,updated=? WHERE id=?",
                    (label,json.dumps(aliases,ensure_ascii=False),code,version,json.dumps(source_ids),stamp,hid))
            else:
                db.execute('INSERT INTO courier_handoffs VALUES(?,?,?,?,?,?,?,?,?,?)',
                    (hid,owner,label,json.dumps(aliases,ensure_ascii=False),code,'pending',version,json.dumps(source_ids),stamp,stamp))
            self.event(db,hid,'owner_authorized_courier_release',owner,dict(version=version,source_ids=source_ids,reason=source_reason))
        record=self.get(hid)
        if record['code']!=code or record['version']!=version:raise RuntimeError('取件码登记读回失败。')
        return record

    def get(self,hid):
        self.ensure_schema()
        if not isinstance(hid,str) or not re.fullmatch(r'[0-9a-f]{32}',hid):raise ValueError('Invalid handoff ID')
        with self.memory.db() as db:r=db.execute('SELECT * FROM courier_handoffs WHERE id=?',(hid,)).fetchone()
        if not r:raise ValueError('退件记录不存在。')
        return dict(r)

    def lookup(self,owner_name,source_text):
        self.ensure_schema()
        if not isinstance(owner_name,str) or not 1<=len(owner_name.strip())<=40:return dict(status='need_owner')
        with self.memory.db() as db:rows=[dict(r) for r in db.execute("SELECT * FROM courier_handoffs WHERE status IN ('pending','announced','completed')")]
        selected=[r for r in rows if owner_name.strip() in json.loads(r['aliases'])]
        if not selected:return dict(status='not_found')
        if len(selected)!=1:return dict(status='ambiguous_owner')
        record=selected[0];aliases=json.loads(record['aliases'])
        if not isinstance(source_text,str) or not any(a in source_text for a in aliases) or not pickup_request(source_text):
            return dict(status='need_owner')
        # Return only the explicitly released handoff, never source chats or a
        # directory of other owners. Spoken code is checked before playback.
        return dict(status='available',handoff_id=record['id'],version=record['version'],
            owner_label=record['owner_label'],code=record['code'],spoken_receipt=public_receipt(record['owner_label'],record['code']))

    def acknowledge(self,hid,version,source_message_id):
        self.ensure_schema()
        with self.memory.db() as db:
            db.execute('BEGIN IMMEDIATE')
            record=db.execute('SELECT * FROM courier_handoffs WHERE id=?',(hid,)).fetchone()
            if not record or record['version']!=version or record['status'] not in ('pending','announced','completed'):
                raise ValueError('取件码记录已更新或关闭，请重新查询。')
            source=db.execute("SELECT role,text FROM messages WHERE id=?",(source_message_id,)).fetchone()
            if not source or source['role']!='user':raise ValueError('需要本轮取件人的确认话语。')
            db.execute("UPDATE courier_handoffs SET status='completed',updated=? WHERE id=?",(now_utc(),hid))
            self.event(db,hid,'recipient_acknowledged_code','courier',dict(version=version,source_message_id=source_message_id,physical_collection_verified=False))
        result=self.get(hid)
        if result['status']!='completed':raise RuntimeError('取件码送达确认读回失败。')
        return dict(handoff_id=hid,delivery_state='acknowledged',physical_collection_verified=False)

    def mark_announced(self,hid,version,response_id):
        with self.memory.db() as db:
            changed=db.execute("UPDATE courier_handoffs SET status='announced',updated=? WHERE id=? AND version=? AND status='pending'",
                (now_utc(),hid,version)).rowcount
            if changed:self.event(db,hid,'pickup_code_spoken','courier',dict(version=version,response_id=response_id,physical_collection_verified=False))

    def cancel(self,hid,owner):
        record=self.get(hid)
        if record['owner']!=owner:raise ValueError('只能由委托人取消自己的退件转告。')
        with self.memory.db() as db:
            db.execute("UPDATE courier_handoffs SET status='cancelled',updated=? WHERE id=?",(now_utc(),hid))
            self.event(db,hid,'owner_cancelled',owner,{})
        return self.get(hid)
