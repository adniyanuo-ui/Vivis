"""Read actual interaction/delivery evidence, never assistant claims of visits."""
from datetime import datetime
from zoneinfo import ZoneInfo

from vivis.interaction.memory import person_key


def visit_evidence(store,actor,person):
    person_key(actor);person_key(person);store.ensure_schema()
    with store.memory.db() as db:
        # Do not reveal a third party's message contents to the caller. A manual
        # closure has no last_announced and is not evidence of audio delivery.
        delivery=db.execute("SELECT MAX(last_announced) FROM robot_tasks WHERE recipient=? AND (creator=? OR recipient=?)",
            (person,actor,actor)).fetchone()[0]
        interaction=db.execute("SELECT MAX(created) FROM messages WHERE person=? AND role='user' AND delivery='user_said'",
            (person,)).fetchone()[0]
    return dict(person_id=person,last_delivery=delivery,last_interaction=interaction,
        currently_present=None,absence_proven=False)


def visit_receipt(evidence,person):
    name=person.get('preferred_address') or person.get('display_name') or '这位同事'
    stamp=evidence.get('last_delivery') or evidence.get('last_interaction')
    if not stamp:return '暂时没有查到能确认'+name+'来过的记录，不能据此说没有来过。'
    dt=datetime.fromisoformat(stamp).astimezone(ZoneInfo('Asia/Shanghai'))
    when=f'{dt.year}年{dt.month}月{dt.day}日{dt.hour}点{dt.minute}分'
    fact=('向'+name+'完整转达过留言') if evidence.get('last_delivery') else ('有与'+name+'的交流记录')
    return '有记录：'+when+'，'+fact+'。这不代表现在仍在场。'
