"""Durable robot task ledger, separate from conversation summaries and motion."""
from datetime import datetime,timezone
import hashlib
import json
import re
import uuid

from vivis.interaction.memory import person_key


STATES=('pending','in_progress','announced','completed','cancelled')
KINDS=('message','reminder','todo')


class MessageMeaningUncertain(ValueError):pass


def validate_config(c):
    terms=c.get('clarification_required_terms',[])
    if (not isinstance(terms,list) or len(terms)>32
            or any(not isinstance(term,str) or not 1<=len(term.strip())<=32 for term in terms)):
        raise ValueError('Invalid message clarification terms')
    if (c.get('schema_version')!='vivis-robot-tasks-1.0' or type(c.get('enabled')) is not bool
            or type(c.get('auto_announce')) is not bool or c.get('timezone')!='Asia/Shanghai'
            or type(c.get('refresh_s')) not in (int,float) or not .5<=c['refresh_s']<=10
            or type(c.get('repeat_after_s')) not in (int,float) or not 30<=c['repeat_after_s']<=86400
            or type(c.get('max_auto_per_visit')) is not int or not 1<=c['max_auto_per_visit']<=5
            or type(c.get('list_limit')) is not int or not 1<=c['list_limit']<=30):raise ValueError('Invalid robot task configuration')


def utc_time(value=None):
    if value in (None,''):return None
    if not isinstance(value,str) or len(value)>40:raise ValueError('Use an ISO8601 time with timezone')
    try:parsed=datetime.fromisoformat(value.replace('Z','+00:00'))
    except ValueError:raise ValueError('Use an ISO8601 time with timezone') from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:raise ValueError('Task time requires timezone')
    return parsed.astimezone(timezone.utc).isoformat()


def now_utc():return datetime.now(timezone.utc).isoformat()


def text(value,limit,required=False):
    if not isinstance(value,str) or len(value)>limit or (required and not value.strip()):raise ValueError('Invalid bounded task text')
    return value.strip()


def relay_matches(content,spoken):
    clean=lambda s:re.sub(r'[^\w]','',s).casefold()
    expected=clean(content)
    return bool(expected) and expected in clean(spoken)


class RobotTasks:
    def __init__(self,memory,config):
        validate_config(config);self.memory=memory;self.c=config;self.ready=False

    def ensure_schema(self):
        if self.ready:return
        with self.memory.db() as db:
            db.executescript('''CREATE TABLE IF NOT EXISTS robot_tasks(
                task_id TEXT PRIMARY KEY,kind TEXT NOT NULL,title TEXT NOT NULL,content TEXT NOT NULL,
                creator TEXT NOT NULL,recipient TEXT,status TEXT NOT NULL,next_action TEXT NOT NULL,
                not_before TEXT,due_at TEXT,created TEXT NOT NULL,updated TEXT NOT NULL,
                source_session TEXT NOT NULL,source_utterance TEXT NOT NULL,source_message_id INTEGER,
                creation_key TEXT UNIQUE NOT NULL,last_announced TEXT,last_attempt TEXT);
                CREATE INDEX IF NOT EXISTS robot_task_inbox ON robot_tasks(recipient,status,not_before);
                CREATE TABLE IF NOT EXISTS message_writes(
                request_id TEXT PRIMARY KEY,actor TEXT NOT NULL,task_id TEXT NOT NULL,action TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS robot_task_events(
                id INTEGER PRIMARY KEY,task_id TEXT NOT NULL,event TEXT NOT NULL,actor TEXT NOT NULL,
                details TEXT NOT NULL,created TEXT NOT NULL);''')
        self.ready=True

    def event(self,db,task_id,kind,actor,details):
        db.execute('INSERT INTO robot_task_events(task_id,event,actor,details,created) VALUES(?,?,?,?,?)',
            (task_id,kind,actor,json.dumps(details,ensure_ascii=False),now_utc()))

    def create(self,args,creator,session,utterance,message_id,known_people,*,source_text='',clarified=False,guard=None):
        self.ensure_schema();person_key(creator)
        allowed={'action','kind','title','content','recipient_id','not_before','due_at','next_action'}
        if not isinstance(args,dict) or not {'action','kind','title','content'}<=args.keys()<=allowed:
            raise ValueError('Task create needs kind,title,content and no unknown fields')
        if args['kind'] not in KINDS:raise ValueError('Unsupported task kind')
        title=text(args['title'],120,True);content=text(args['content'],500,True)
        if not clarified:
            for term in self.c.get('clarification_required_terms',[]):
                if re.search(r'(?<![a-z0-9])'+re.escape(term.casefold())+r'(?![a-z0-9])',(source_text+' '+content).casefold()):
                    raise MessageMeaningUncertain('留言含尚未澄清的识别词，请先问清楚；没有保存，也不能猜测替换。')
        if not re.search(r'\w',content):raise ValueError('Task requires actual words')
        recipient=args.get('recipient_id','')
        if not isinstance(recipient,str):raise ValueError('Recipient ID must be a registered string')
        recipient=recipient or None
        if recipient is not None and recipient not in known_people:raise ValueError('Choose a registered recipient from get_robot_status.known_people')
        if args['kind'] in ('message','reminder') and not recipient:raise ValueError('Message/reminder needs an explicit registered recipient')
        start=utc_time(args.get('not_before'));due=utc_time(args.get('due_at'))
        if start and due and due<start:raise ValueError('Due time is before task start')
        if not isinstance(utterance,str) or not 1<=len(utterance)<=128:raise ValueError('Task requires a fresh user utterance')
        if message_id is not None and (type(message_id) is not int or message_id<1):raise ValueError('Invalid task source message')
        next_action=text(args.get('next_action','等待收件人出现并转告' if recipient else '等待后续安排'),300)
        key=hashlib.sha256(json.dumps([creator,str(session),utterance],ensure_ascii=False).encode()).hexdigest()
        stamp=now_utc();task_id=uuid.uuid4().hex
        with self.memory.db() as db:
            db.execute('BEGIN IMMEDIATE')
            if guard:guard()
            prior=db.execute('SELECT * FROM robot_tasks WHERE creator=? AND source_session=? AND source_utterance=? ORDER BY created LIMIT 1',
                (creator,str(session),utterance)).fetchone()
            if prior:return dict(prior,effect='already_acknowledged' if prior['status']=='completed' else 'unchanged')
            db.execute('INSERT INTO robot_tasks(task_id,kind,title,content,creator,recipient,status,next_action,not_before,due_at,created,updated,source_session,source_utterance,source_message_id,creation_key) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (task_id,args['kind'],title,content,creator,recipient,'pending',next_action,start,due,stamp,stamp,str(session)[:100],utterance,message_id,key))
            self.event(db,task_id,'created',creator,dict(source_message_id=message_id,source_utterance=utterance))
            db.execute('INSERT INTO message_writes VALUES(?,?,?,?)',(key,creator,task_id,'create'))
            if guard:guard()
        self.memory.version+=1
        verified=self.get(task_id)
        if verified['content']!=content or verified['recipient']!=recipient:raise RuntimeError('Message readback did not match its write')
        return dict(verified,effect='created')

    def supplement(self,task_id,actor,session,utterance,message_id,addition,guard=None):
        """Only append an explicitly requested supplement to an exact owned ID."""
        self.ensure_schema();person_key(actor);addition=text(addition,300,True)
        key=hashlib.sha256(json.dumps([actor,str(session),utterance],ensure_ascii=False).encode()).hexdigest()
        with self.memory.db() as db:
            db.execute('BEGIN IMMEDIATE')
            if guard:guard()
            prior=db.execute('SELECT task_id FROM message_writes WHERE request_id=?',(key,)).fetchone()
            if prior:
                row=db.execute('SELECT * FROM robot_tasks WHERE task_id=?',(prior['task_id'],)).fetchone()
                return dict(row,effect='unchanged')
            row=db.execute('SELECT * FROM robot_tasks WHERE task_id=?',(task_id,)).fetchone()
            if not row or row['creator']!=actor:raise ValueError('只能补充本人明确指定的留言。')
            if row['status'] not in ('pending','announced'):raise ValueError('该留言已经关闭，请作为新留言交代。')
            content=row['content']+' 补充：'+addition
            if len(content)>500:raise ValueError('补充后留言过长，请分成另一条留言。')
            db.execute("UPDATE robot_tasks SET content=?,status='pending',last_attempt=NULL,updated=? WHERE task_id=?",(content,now_utc(),task_id))
            self.event(db,task_id,'supplemented',actor,dict(source_session=session,source_utterance=utterance,source_message_id=message_id,addition=addition))
            db.execute('INSERT INTO message_writes VALUES(?,?,?,?)',(key,actor,task_id,'supplement'))
            if guard:guard()
        self.memory.version+=1
        return dict(self.get(task_id),effect='merged')

    def get(self,task_id):
        self.ensure_schema()
        if not isinstance(task_id,str) or not re.fullmatch(r'[a-f0-9]{32}',task_id):raise ValueError('Invalid task ID')
        with self.memory.db() as db:row=db.execute('SELECT * FROM robot_tasks WHERE task_id=?',(task_id,)).fetchone()
        if not row:raise ValueError('Unknown task')
        return dict(row)

    def list(self,status='active',recipient=None,after_task_id=None,limit=None,visible_to=None,content_keyword=None):
        self.ensure_schema()
        if status not in (*STATES,'active','all'):raise ValueError('Unknown task status')
        if recipient:person_key(recipient)
        clauses=[];params=[]
        if visible_to is not None:
            person_key(visible_to)
            clauses.append("(kind='todo' OR creator=? OR recipient=?)");params.extend([visible_to,visible_to])
        if status=='active':clauses.append("(status IN ('pending','in_progress') OR (kind='todo' AND status='announced'))")
        elif status=='announced':clauses.append("(status='announced' OR (kind IN ('message','reminder') AND status='completed'))")
        elif status!='all':clauses.append('status=?');params.append(status)
        if recipient:clauses.append('recipient=?');params.append(recipient)
        if content_keyword:
            if not isinstance(content_keyword,str) or not 2<=len(content_keyword)<=40:raise ValueError('Invalid message keyword')
            clauses.append('instr(content,?)>0');params.append(content_keyword)
        if after_task_id:
            cursor=self.get(after_task_id)
            clauses.append('(created>? OR (created=? AND task_id>?))');params.extend([cursor['created'],cursor['created'],cursor['task_id']])
        where=' WHERE '+' AND '.join(clauses) if clauses else ''
        with self.memory.db() as db:
            rows=[dict(r) for r in db.execute('SELECT * FROM robot_tasks'+where+' ORDER BY created,task_id LIMIT?',[*params,limit or self.c['list_limit']])]
        return rows

    def page(self,status='active',recipient=None,after_task_id=None,visible_to=None,content_keyword=None):
        rows=self.list(status,recipient,after_task_id,self.c['list_limit']+1,visible_to,content_keyword);more=len(rows)>self.c['list_limit'];rows=rows[:self.c['list_limit']]
        return dict(tasks=rows,has_more=more,next_after_task_id=rows[-1]['task_id'] if rows and more else None,counts=self.counts(visible_to))

    def counts(self,visible_to=None):
        self.ensure_schema()
        where='';params=[]
        if visible_to is not None:
            person_key(visible_to)
            where=" WHERE kind='todo' OR creator=? OR recipient=?";params=[visible_to,visible_to]
        with self.memory.db() as db:return {r['status']:r['n'] for r in db.execute('SELECT status,COUNT(*) n FROM robot_tasks'+where+' GROUP BY status',params)}

    def update(self,args,actor,utterance,guard=None):
        if not isinstance(args,dict) or not {'action','task_id'}<=args.keys()<={'action','task_id','status','next_action'}:
            raise ValueError('Task update fields are invalid')
        task=self.get(args['task_id']);person_key(actor)
        if actor not in (task['creator'],task['recipient']):raise ValueError('Only requester or recipient may update this task')
        if not utterance:raise ValueError('Task update requires fresh user confirmation')
        status=args.get('status',task['status'])
        if status not in ('pending','in_progress','completed','cancelled') and not (status=='announced' and 'status' not in args):
            raise ValueError('announced is recorded only by actual playback')
        if task['status'] in ('completed','cancelled') and status!=task['status']:raise ValueError('Closed task cannot silently reopen; create a new task')
        if status=='completed' and task['kind'] in ('message','reminder') and actor!=task['recipient']:
            raise ValueError('Only the recipient can confirm receipt; speaking is not acknowledgement')
        next_action=text(args.get('next_action',task['next_action']),300)
        with self.memory.db() as db:
            if guard:guard()
            db.execute('UPDATE robot_tasks SET status=?,next_action=?,updated=? WHERE task_id=?',(status,next_action,now_utc(),task['task_id']))
            self.event(db,task['task_id'],'user_update',actor,dict(status=status,source_utterance=utterance,
                basis='recipient_acknowledgement' if status=='completed' and task['kind']!='todo' else 'user_report'))
            if guard:guard()
        self.memory.version+=1;return self.get(task['task_id'])

    def acknowledge(self,task_id,recipient,utterance,guard=None):
        task=self.get(task_id)
        if task['kind'] not in ('message','reminder'):raise ValueError('Only message delivery can be acknowledged')
        result=self.update(dict(action='update',task_id=task_id,status='completed',next_action='收件人已确认收到留言；不跟踪所述事项的实际执行'),recipient,utterance,guard=guard)
        if result['status']!='completed':raise RuntimeError('Message acknowledgement readback failed')
        return dict(result,delivery_state='acknowledged',physical_execution_tracked=False)

    def due(self,recipient,stamp=None):
        self.ensure_schema();person_key(recipient);stamp=utc_time(stamp) if stamp else now_utc()
        cutoff=datetime.fromisoformat(stamp).timestamp()-self.c['repeat_after_s']
        cutoff=datetime.fromtimestamp(cutoff,timezone.utc).isoformat()
        with self.memory.db() as db:
            return [dict(r) for r in db.execute("SELECT * FROM robot_tasks WHERE recipient=? AND kind IN ('message','reminder') AND status IN ('pending','in_progress') AND (not_before IS NULL OR not_before<=?) AND (last_attempt IS NULL OR last_attempt<=?) ORDER BY COALESCE(due_at,not_before,created),created LIMIT?",(recipient,stamp,cutoff,self.c['list_limit']))]

    def prepare_relay(self,task_id,recipient):
        task=self.get(task_id)
        if task['recipient']!=recipient or task['kind']=='todo':raise ValueError('This task is not a relay for the current recipient')
        if task['status'] in ('completed','cancelled') or (task['not_before'] and task['not_before']>now_utc()):raise ValueError('Task is not due or already closed')
        with self.memory.db() as db:
            db.execute('UPDATE robot_tasks SET last_attempt=? WHERE task_id=?',(now_utc(),task_id))
            self.event(db,task_id,'relay_prepared',recipient,{})
        return task

    def mark_announced(self,task_id,recipient,response_id,guard=None):
        task=self.get(task_id)
        if task['recipient']!=recipient:raise ValueError('Wrong relay recipient')
        with self.memory.db() as db:
            db.execute('BEGIN IMMEDIATE')
            if guard:guard()
            cursor=db.execute("UPDATE robot_tasks SET status='announced',next_action=?,last_announced=?,updated=? WHERE task_id=? AND status IN ('pending','in_progress','announced')",('已完整转达；无需口头确认，不跟踪事情执行',now_utc(),now_utc(),task_id))
            if cursor.rowcount:self.event(db,task_id,'audio_played',recipient,dict(response_id=response_id,delivery_complete=True,understanding_confirmed=False))
            if guard:guard()
        self.memory.version+=1
        result=self.get(task_id)
        if result['status'] not in ('announced','completed'):raise ValueError('Message changed before delivery could be recorded')
        return result

    def bind_source(self,session,utterance,message_id):
        self.ensure_schema()
        with self.memory.db() as db:db.execute('UPDATE robot_tasks SET source_message_id=? WHERE source_session=? AND source_utterance=? AND source_message_id IS NULL',(message_id,session,utterance))
