"""Durable robot task ledger, separate from conversation summaries and motion."""
from datetime import datetime,timezone
import hashlib
import json
import re
import uuid

from vivis.interaction.memory import person_key


STATES=('pending','in_progress','announced','completed','cancelled')
KINDS=('message','reminder','todo')


class ReconciliationRequired(ValueError):
    def __init__(self,candidates,has_more=False):
        super().__init__('先按语义核对同一委托：复述或补充用merge_task_id，确为另一件事才用distinct_from；不能直接重复建单。')
        self.candidates=candidates;self.has_more=has_more


class MessageMeaningUncertain(ValueError):pass


def normalized_content(value):return re.sub(r'[^\w]','',value).casefold()


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
                CREATE TABLE IF NOT EXISTS robot_task_events(
                id INTEGER PRIMARY KEY,task_id TEXT NOT NULL,event TEXT NOT NULL,actor TEXT NOT NULL,
                details TEXT NOT NULL,created TEXT NOT NULL);''')
        self.ready=True

    def event(self,db,task_id,kind,actor,details):
        db.execute('INSERT INTO robot_task_events(task_id,event,actor,details,created) VALUES(?,?,?,?,?)',
            (task_id,kind,actor,json.dumps(details,ensure_ascii=False),now_utc()))

    def create(self,args,creator,session,utterance,message_id,known_people,*,source_text='',clarified=False):
        self.ensure_schema();person_key(creator)
        allowed={'action','kind','title','content','recipient_id','not_before','due_at','next_action',
                 'merge_task_id','distinct_from','merge_reason'}
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
        merge=args.get('merge_task_id');distinct=args.get('distinct_from',[])
        if merge is not None and (not isinstance(merge,str) or not re.fullmatch(r'[a-f0-9]{32}',merge)):
            raise ValueError('Invalid merge message ID')
        if (not isinstance(distinct,list) or len(distinct)>25
                or any(not isinstance(v,str) or not re.fullmatch(r'[a-f0-9]{32}',v) for v in distinct)):
            raise ValueError('Invalid reviewed message IDs')
        reason=text(args.get('merge_reason',''),300)
        if not reason and (merge or distinct):
            reason='调用方在当前话轮明确选择'+('合并已有留言' if merge else '不同委托另建留言')+'；具体依据保留于源话轮和内容审计。'
        canonical=[creator,str(session),utterance,args['kind'],content,recipient,start,due]
        key=hashlib.sha256(json.dumps(canonical,ensure_ascii=False).encode()).hexdigest()
        stamp=now_utc();task_id=uuid.uuid4().hex
        with self.memory.db() as db:
            db.execute('BEGIN IMMEDIATE')
            prior=db.execute('SELECT * FROM robot_tasks WHERE creation_key=?',(key,)).fetchone()
            if prior:return dict(prior,effect='unchanged')
            if args['kind'] in ('message','reminder'):
                existing_rows=[dict(r) for r in db.execute("SELECT * FROM robot_tasks WHERE creator=? AND recipient=? AND kind=? AND status!='cancelled' ORDER BY created DESC",
                    (creator,recipient,args['kind']))]
                for existing in existing_rows:
                    if (normalized_content(existing['content'])==normalized_content(content)
                            and existing['not_before']==start and existing['due_at']==due):
                        return dict(existing,effect='already_acknowledged' if existing['status']=='completed' else 'unchanged')
                active=[r for r in existing_rows if r['status']!='completed']
                candidates=active[:20]+[r for r in existing_rows if r['status']=='completed'][:5]
                if merge:
                    existing=db.execute('SELECT * FROM robot_tasks WHERE task_id=?',(merge,)).fetchone()
                    if (not existing or existing['creator']!=creator or existing['recipient']!=recipient
                            or existing['kind']!=args['kind'] or existing['status']=='cancelled'):
                        raise ValueError('Can only merge your own message to the same recipient and kind')
                    if existing['status']=='completed':return dict(existing,effect='already_acknowledged')
                    changed=(existing['content']!=content or existing['not_before']!=start or existing['due_at']!=due)
                    db.execute('UPDATE robot_tasks SET title=?,content=?,not_before=?,due_at=?,next_action=?,status=?,last_attempt=?,updated=? WHERE task_id=?',
                        (title,content,start,due,next_action,'pending' if changed else existing['status'],
                         None if changed else existing['last_attempt'],stamp,merge))
                    self.event(db,merge,'message_reconciled',creator,dict(reason=reason,
                        previous_content=existing['content'],content=content,source_session=str(session),
                        source_utterance=utterance,source_message_id=message_id))
                    task_id=merge
                elif candidates and (len(active)>20 or not {r['task_id'] for r in candidates}<=set(distinct)):
                    fields=('task_id','title','content','status','not_before','due_at')
                    raise ReconciliationRequired([{k:r[k] for k in fields} for r in candidates],len(active)>20)
                if merge:
                    result=dict(db.execute('SELECT * FROM robot_tasks WHERE task_id=?',(task_id,)).fetchone())
                    result['effect']='merged'
                else:result=None
            else:result=None
            if result is None:
                db.execute('INSERT INTO robot_tasks(task_id,kind,title,content,creator,recipient,status,next_action,not_before,due_at,created,updated,source_session,source_utterance,source_message_id,creation_key) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (task_id,args['kind'],title,content,creator,recipient,'pending',next_action,start,due,stamp,stamp,str(session)[:100],utterance,message_id,key))
                self.event(db,task_id,'created',creator,dict(source_message_id=message_id,source_utterance=utterance,
                    distinct_from=distinct,semantic_reason=reason))
        self.memory.version+=1
        verified=self.get(task_id)
        if verified['content']!=content or verified['recipient']!=recipient:raise RuntimeError('Message readback did not match its write')
        return dict(verified,effect='merged' if merge else 'created')

    def get(self,task_id):
        self.ensure_schema()
        if not isinstance(task_id,str) or not re.fullmatch(r'[a-f0-9]{32}',task_id):raise ValueError('Invalid task ID')
        with self.memory.db() as db:row=db.execute('SELECT * FROM robot_tasks WHERE task_id=?',(task_id,)).fetchone()
        if not row:raise ValueError('Unknown task')
        return dict(row)

    def list(self,status='active',recipient=None,after_task_id=None,limit=None,visible_to=None):
        self.ensure_schema()
        if status not in (*STATES,'active','all'):raise ValueError('Unknown task status')
        if recipient:person_key(recipient)
        clauses=[];params=[]
        if visible_to is not None:
            person_key(visible_to)
            clauses.append("(kind='todo' OR creator=? OR recipient=?)");params.extend([visible_to,visible_to])
        if status=='active':clauses.append("status IN ('pending','in_progress','announced')")
        elif status!='all':clauses.append('status=?');params.append(status)
        if recipient:clauses.append('recipient=?');params.append(recipient)
        if after_task_id:
            cursor=self.get(after_task_id)
            clauses.append('(created>? OR (created=? AND task_id>?))');params.extend([cursor['created'],cursor['created'],cursor['task_id']])
        where=' WHERE '+' AND '.join(clauses) if clauses else ''
        with self.memory.db() as db:
            rows=[dict(r) for r in db.execute('SELECT * FROM robot_tasks'+where+' ORDER BY created,task_id LIMIT?',[*params,limit or self.c['list_limit']])]
        return rows

    def page(self,status='active',recipient=None,after_task_id=None,visible_to=None):
        rows=self.list(status,recipient,after_task_id,self.c['list_limit']+1,visible_to);more=len(rows)>self.c['list_limit'];rows=rows[:self.c['list_limit']]
        return dict(tasks=rows,has_more=more,next_after_task_id=rows[-1]['task_id'] if rows and more else None,counts=self.counts(visible_to))

    def counts(self,visible_to=None):
        self.ensure_schema()
        where='';params=[]
        if visible_to is not None:
            person_key(visible_to)
            where=" WHERE kind='todo' OR creator=? OR recipient=?";params=[visible_to,visible_to]
        with self.memory.db() as db:return {r['status']:r['n'] for r in db.execute('SELECT status,COUNT(*) n FROM robot_tasks'+where+' GROUP BY status',params)}

    def update(self,args,actor,utterance):
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
            db.execute('UPDATE robot_tasks SET status=?,next_action=?,updated=? WHERE task_id=?',(status,next_action,now_utc(),task['task_id']))
            self.event(db,task['task_id'],'user_update',actor,dict(status=status,source_utterance=utterance,
                basis='recipient_acknowledgement' if status=='completed' and task['kind']!='todo' else 'user_report'))
        self.memory.version+=1;return self.get(task['task_id'])

    def acknowledge(self,task_id,recipient,utterance):
        task=self.get(task_id)
        if task['kind'] not in ('message','reminder'):raise ValueError('Only message delivery can be acknowledged')
        result=self.update(dict(action='update',task_id=task_id,status='completed',next_action='收件人已确认收到留言；不跟踪所述事项的实际执行'),recipient,utterance)
        if result['status']!='completed':raise RuntimeError('Message acknowledgement readback failed')
        return dict(result,delivery_state='acknowledged',physical_execution_tracked=False)

    def due(self,recipient,stamp=None):
        self.ensure_schema();person_key(recipient);stamp=utc_time(stamp) if stamp else now_utc()
        cutoff=datetime.fromisoformat(stamp).timestamp()-self.c['repeat_after_s']
        cutoff=datetime.fromtimestamp(cutoff,timezone.utc).isoformat()
        with self.memory.db() as db:
            return [dict(r) for r in db.execute("SELECT * FROM robot_tasks WHERE recipient=? AND kind IN ('message','reminder') AND status IN ('pending','in_progress','announced') AND (not_before IS NULL OR not_before<=?) AND (last_attempt IS NULL OR last_attempt<=?) ORDER BY COALESCE(due_at,not_before,created),created LIMIT?",(recipient,stamp,cutoff,self.c['list_limit']))]

    def prepare_relay(self,task_id,recipient):
        task=self.get(task_id)
        if task['recipient']!=recipient or task['kind']=='todo':raise ValueError('This task is not a relay for the current recipient')
        if task['status'] in ('completed','cancelled') or (task['not_before'] and task['not_before']>now_utc()):raise ValueError('Task is not due or already closed')
        with self.memory.db() as db:
            db.execute('UPDATE robot_tasks SET last_attempt=? WHERE task_id=?',(now_utc(),task_id))
            self.event(db,task_id,'relay_prepared',recipient,{})
        return task

    def mark_announced(self,task_id,recipient,response_id):
        task=self.get(task_id)
        if task['recipient']!=recipient:raise ValueError('Wrong relay recipient')
        with self.memory.db() as db:
            cursor=db.execute("UPDATE robot_tasks SET status='announced',last_announced=?,updated=? WHERE task_id=? AND status IN ('pending','in_progress','announced')",(now_utc(),now_utc(),task_id))
            if cursor.rowcount:self.event(db,task_id,'audio_played',recipient,dict(response_id=response_id,understanding_confirmed=False))
        self.memory.version+=1

    def bind_source(self,session,utterance,message_id):
        self.ensure_schema()
        with self.memory.db() as db:db.execute('UPDATE robot_tasks SET source_message_id=? WHERE source_session=? AND source_utterance=? AND source_message_id IS NULL',(message_id,session,utterance))
