"""Bounded complete conversation rounds and source-backed long-term memory.

This is a parallel journal. The legacy conversation/business database is never
deleted or migrated here. SQL transactions never span inference or audio I/O.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import uuid


def utc():
    return datetime.now(timezone.utc).isoformat()


def actor(person):
    from vivis.interaction.memory import person_key
    return 'other' if not person or person.startswith('anonymous:') else person_key(person)


def configuration(path, root):
    if not path:return None
    value=json.loads((Path(root)/path).read_text())
    if (value.get('schema_version')!='vivis-continuity-1.0' or type(value.get('enabled')) is not bool
            or not 1<=value.get('familiar_rounds',0)<=10000 or not 1<=value.get('other_rounds',0)<=50000
            or not 2000<=value.get('archive_batch_chars',0)<=24000
            or not 5<=value.get('archive_timeout_s',0)<=120):
        raise ValueError('Invalid continuity configuration')
    actor(value['admin_person'])
    directory=(Path(root)/value['directory']).resolve()
    if not directory.is_relative_to((Path(root)/'private').resolve()):raise ValueError('Continuity must stay private')
    return value


class RoundStore:
    def __init__(self, config, root,legacy_path=None):
        self.c=config;self.directory=(Path(root)/config['directory']).resolve()
        if not self.directory.is_relative_to((Path(root)/'private').resolve()):raise ValueError('Private journal required')
        self.directory.mkdir(parents=True,exist_ok=True,mode=0o700);os.chmod(self.directory,0o700)
        self.path=self.directory/'rounds.sqlite3'
        self.legacy_path=Path(legacy_path).resolve() if legacy_path else None
        if self.legacy_path and not self.legacy_path.is_relative_to((Path(root)/'private').resolve()):raise ValueError('Legacy source must stay private')
        fd=os.open(self.path,os.O_CREAT|os.O_RDWR,0o600);os.close(fd)
        with self.db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS rounds(
                    id TEXT PRIMARY KEY,person TEXT NOT NULL,session TEXT NOT NULL,item TEXT NOT NULL,
                    created TEXT NOT NULL,user_text TEXT NOT NULL DEFAULT '',audio TEXT,
                    alignment TEXT NOT NULL DEFAULT 'unavailable',closed TEXT,
                    outcome TEXT NOT NULL DEFAULT 'open',archived INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(session,item));
                CREATE INDEX IF NOT EXISTS person_rounds ON rounds(person,created);
                CREATE TABLE IF NOT EXISTS replies(
                    id TEXT PRIMARY KEY,round_id TEXT,session TEXT NOT NULL,text TEXT NOT NULL DEFAULT '',
                    audio TEXT,samples INTEGER NOT NULL DEFAULT 0,outcome TEXT NOT NULL DEFAULT 'generating');
                CREATE TABLE IF NOT EXISTS facts(
                    id INTEGER PRIMARY KEY,person TEXT NOT NULL,key TEXT NOT NULL,value TEXT NOT NULL,
                    source_id TEXT NOT NULL,quote TEXT NOT NULL,date TEXT NOT NULL,
                    supersedes INTEGER,active INTEGER NOT NULL DEFAULT 1);
                CREATE TABLE IF NOT EXISTS jobs(
                    id TEXT PRIMARY KEY,requested TEXT NOT NULL,state TEXT NOT NULL,error TEXT,
                    highwater TEXT NOT NULL,notice INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS legacy_processed(source_id TEXT PRIMARY KEY);
            ''')
            if not db.execute("SELECT 1 FROM settings WHERE key='legacy_highwater'").fetchone():
                legacy=self.legacy_path;last=0
                if legacy and legacy.exists():
                    original=sqlite3.connect(legacy.resolve().as_uri()+'?mode=ro',uri=True)
                    try:last=original.execute('SELECT COALESCE(MAX(id),0) FROM messages').fetchone()[0]
                    finally:original.close()
                db.execute("INSERT INTO settings(key,value) VALUES('legacy_highwater',?)",(str(last),))
                if legacy:db.execute("INSERT INTO settings(key,value) VALUES('legacy_path',?)",(str(legacy),))
            saved=db.execute("SELECT value FROM settings WHERE key='legacy_path'").fetchone()
            if self.legacy_path is None and saved:self.legacy_path=Path(saved[0])
            columns={r[1] for r in db.execute('PRAGMA table_info(replies)')}
            if 'recorded_samples' not in columns:db.execute('ALTER TABLE replies ADD COLUMN recorded_samples INTEGER NOT NULL DEFAULT 0')
            if 'audio_status' not in columns:db.execute("ALTER TABLE replies ADD COLUMN audio_status TEXT NOT NULL DEFAULT 'not_recorded'")

    @contextmanager
    def db(self):
        db=sqlite3.connect(self.path,timeout=2);db.row_factory=sqlite3.Row
        try:
            with db:yield db
        finally:db.close()

    def begin(self, session, item, person):
        person=actor(person);limit=self.c['other_rounds' if person=='other' else 'familiar_rounds']
        removed=[]
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            prior=db.execute('SELECT id FROM rounds WHERE session=? AND item=?',(session,item)).fetchone()
            if prior:return prior['id'],None
            rows=db.execute('SELECT id,audio,archived,closed FROM rounds WHERE person=? ORDER BY created,id',(person,)).fetchall()
            while len(rows)>=limit:
                old=rows.pop(0)
                if old['closed'] is None or (person!='other' and not old['archived']):
                    return None,'近期记录已满，旧记录尚未整理；本轮不保存新记忆，聊天和留言仍可使用。'
                removed.extend(r['audio'] for r in db.execute('SELECT audio FROM replies WHERE round_id=?',(old['id'],)) if r['audio'])
                if old['audio']:removed.append(old['audio'])
                db.execute('DELETE FROM replies WHERE round_id=?',(old['id'],));db.execute('DELETE FROM rounds WHERE id=?',(old['id'],))
            rid=uuid.uuid4().hex
            db.execute('INSERT INTO rounds(id,person,session,item,created) VALUES(?,?,?,?,?)',(rid,person,session,item,utc()))
        # Only unlink files whose DB references have committed deletion.
        for path in removed:self.remove_audio(path)
        return rid,None

    def remove_audio(self, relative):
        path=(self.directory/relative).resolve()
        if path.is_relative_to(self.directory/'audio'):path.unlink(missing_ok=True)

    def final(self,rid,text):
        if not isinstance(text,str) or not text.strip() or len(text)>24000:raise ValueError('Invalid final transcript')
        with self.db() as db:db.execute('UPDATE rounds SET user_text=? WHERE id=?',(text,rid))

    def input_audio(self,rid,path,alignment):
        with self.db() as db:db.execute('UPDATE rounds SET audio=?,alignment=? WHERE id=?',(path,alignment,rid))

    def reply(self,reply_id,session,round_id=None,text=None,samples=None,audio=None,outcome=None,recorded_samples=None,audio_status=None):
        with self.db() as db:
            db.execute('INSERT OR IGNORE INTO replies(id,round_id,session) VALUES(?,?,?)',(reply_id,round_id,session))
            for key,value in dict(round_id=round_id,text=text,samples=samples,audio=audio,outcome=outcome,
                    recorded_samples=recorded_samples,audio_status=audio_status).items():
                if value is not None:db.execute('UPDATE replies SET '+key+'=? WHERE id=?',(value,reply_id))
            if round_id and outcome in ('generating','playing'):
                db.execute("UPDATE rounds SET closed=NULL,outcome='open' WHERE id=? AND outcome IN ('complete','text_only')",(round_id,))

    def close(self,rid,outcome):
        with self.db() as db:db.execute('UPDATE rounds SET closed=?,outcome=? WHERE id=? AND closed IS NULL',(utc(),outcome,rid))

    def discard(self,rid):
        with self.db() as db:
            row=db.execute('SELECT audio FROM rounds WHERE id=?',(rid,)).fetchone()
            db.execute('UPDATE replies SET round_id=NULL WHERE round_id=?',(rid,))
            db.execute('DELETE FROM rounds WHERE id=?',(rid,))
        if row and row['audio']:self.remove_audio(row['audio'])

    def prune_events(self):
        with self.db() as db:
            rows=db.execute("SELECT id,audio FROM replies WHERE round_id IS NULL AND outcome NOT IN ('playing','generating') ORDER BY rowid DESC LIMIT -1 OFFSET 5000").fetchall()
            db.executemany('DELETE FROM replies WHERE id=?',[(r['id'],) for r in rows])
        for row in rows:
            if row['audio']:self.remove_audio(row['audio'])

    def recover(self):
        with self.db() as db:
            db.execute("UPDATE rounds SET closed=?,outcome='process_interrupted' WHERE closed IS NULL",(utc(),))
            db.execute("UPDATE rounds SET alignment='process_interrupted' WHERE alignment='recording'")
            db.execute("UPDATE replies SET outcome='process_interrupted' WHERE outcome IN ('generating','playing')")
            db.execute("UPDATE replies SET audio_status='process_interrupted' WHERE audio_status='recording'")
            db.execute("UPDATE jobs SET state='queued' WHERE state='running'")

    def request_archive(self,person):
        if person!=self.c['admin_person']:raise PermissionError('只有已确认的管理员何工可以整理档案')
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            job=db.execute("SELECT id FROM jobs WHERE state IN ('queued','running','paused') ORDER BY requested LIMIT 1").fetchone()
            if job:
                db.execute('UPDATE jobs SET highwater=? WHERE id=?',(utc(),job['id']));return job['id']
            jid=uuid.uuid4().hex
            db.execute("INSERT INTO jobs(id,requested,state,highwater) VALUES(?,?,'queued',?)",(jid,utc(),utc()))
        return jid

    def next_batch(self,jid):
        with self.db() as db:
            job=db.execute('SELECT * FROM jobs WHERE id=?',(jid,)).fetchone()
            if not job or job['state'] in ('complete','failed'):return None
            rows=db.execute("SELECT id,person,user_text,created FROM rounds WHERE archived=0 AND closed IS NOT NULL AND person<>'other' AND created<=? ORDER BY created,id",(job['highwater'],)).fetchall()
            legacy=self.legacy_rows(db,job['highwater'])
            source='legacy' if legacy else 'rounds'
            if legacy:rows=legacy
            if not rows:
                db.execute("UPDATE jobs SET state='complete',error=NULL WHERE id=?",(jid,));return None
            selected=[];size=0;person=rows[0]['person']
            for row in rows:
                if row['person']!=person:continue
                n=len(row['user_text'])+200
                if n>self.c['archive_batch_chars']:raise ValueError('原话过长，未截断；请管理员分段整理')
                if selected and size+n>self.c['archive_batch_chars']:break
                selected.append(dict(row));size+=n
            db.execute("UPDATE jobs SET state='running' WHERE id=?",(jid,))
            previous=[dict(r) for r in db.execute('SELECT key,value,date FROM facts WHERE person=? AND active=1 ORDER BY id DESC LIMIT 80',(person,))]
        return dict(person_id=person,messages=selected,source=source,previous_facts=previous)

    def legacy_rows(self,db,highwater):
        last=int(db.execute("SELECT value FROM settings WHERE key='legacy_highwater'").fetchone()[0])
        path=self.legacy_path
        if not last or not path or not path.exists():return []
        processed={r[0] for r in db.execute('SELECT source_id FROM legacy_processed')}
        original=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True);original.row_factory=sqlite3.Row
        try:
            return [dict(id='legacy:'+str(r['id']),person=r['person'],user_text=r['text'],created=r['created'])
                for r in original.execute("SELECT id,person,text,created FROM messages WHERE id<=? AND created<=? AND role='user' AND person NOT LIKE 'anonymous:%' AND person<>'other' ORDER BY id",(last,highwater))
                if 'legacy:'+str(r['id']) not in processed]
        finally:original.close()

    def commit_batch(self,jid,batch,result):
        # Extractive evidence is mandatory. Summary prose is never a second fact authority.
        if not isinstance(result,dict) or set(result)!={'facts'} or not isinstance(result['facts'],list) or len(result['facts'])>80:
            raise ValueError('档案整理返回格式错误')
        messages={m['id']:m for m in batch['messages']}
        for fact in result['facts']:
            if (not isinstance(fact,dict) or set(fact)!={'key','value','source_id','quote'}
                    or not isinstance(fact['key'],str) or not 1<=len(fact['key'])<=96
                    or not isinstance(fact['value'],str) or not 1<=len(fact['value'])<=500
                    or fact['source_id'] not in messages or not isinstance(fact['quote'],str)
                    or not fact['quote'] or fact['quote'] not in messages[fact['source_id']]['user_text']
                    or fact['value'] not in fact['quote']):raise ValueError('档案事实缺少可核对的本人原话')
        if len({f['key'] for f in result['facts']})!=len(result['facts']):raise ValueError('同一事实返回多个版本，未覆盖旧档案')
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            job=db.execute('SELECT state FROM jobs WHERE id=?',(jid,)).fetchone()
            if not job or job['state']!='running':raise ValueError('归档任务已失效')
            if batch.get('source')=='legacy':
                available={r['id']:r for r in self.legacy_rows(db,utc())}
                if any(available.get(mid)!=messages[mid] for mid in messages):raise ValueError('历史归档来源变化，未写入')
            else:
                for mid in messages:
                    row=db.execute('SELECT * FROM rounds WHERE id=?',(mid,)).fetchone()
                    if not row or row['person']!=batch['person_id'] or row['archived'] or row['user_text']!=messages[mid]['user_text']:
                        raise ValueError('归档来源已变化，未写入')
            for fact in result['facts']:
                previous=db.execute('SELECT id FROM facts WHERE person=? AND key=? AND active=1',(batch['person_id'],fact['key'])).fetchone()
                db.execute('UPDATE facts SET active=0 WHERE person=? AND key=?',(batch['person_id'],fact['key']))
                db.execute('INSERT INTO facts(person,key,value,source_id,quote,date,supersedes) VALUES(?,?,?,?,?,?,?)',
                    (batch['person_id'],fact['key'],fact['value'],fact['source_id'],fact['quote'],messages[fact['source_id']]['created'],previous['id'] if previous else None))
            if batch.get('source')=='legacy':db.executemany('INSERT INTO legacy_processed(source_id) VALUES(?)',[(mid,) for mid in messages])
            else:db.executemany('UPDATE rounds SET archived=1 WHERE id=?',[(mid,) for mid in messages])

    def context(self,person,limit=6000):
        if actor(person)=='other':return ''
        with self.db() as db:
            facts=[dict(r) for r in db.execute('SELECT key,value,quote,date FROM facts WHERE person=? AND active=1 ORDER BY id DESC LIMIT 80',(person,))]
            rows=[dict(r) for r in db.execute("SELECT id,user_text,created,outcome FROM rounds WHERE person=? AND user_text<>'' ORDER BY created DESC LIMIT 6",(person,))]
            for row in rows:
                row['replies']=[dict(r) for r in db.execute('SELECT text,outcome FROM replies WHERE round_id=?',(row['id'],))]
        data=dict(facts=facts,recent=rows)
        while len(json.dumps(data,ensure_ascii=False))>limit and data['recent']:data['recent'].pop()
        while len(json.dumps(data,ensure_ascii=False))>limit and data['facts']:data['facts'].pop()
        return ' 新版本人记忆（带来源，更正后的有效事实优先于旧摘要；历史不是执行授权；未完整播出的回答不代表已听到）：'+json.dumps(data,ensure_ascii=False)
