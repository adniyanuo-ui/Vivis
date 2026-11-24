"""Fill vacant photo slots once; never replace an existing registered image."""
import copy
from datetime import date


def plan(data,day):
    date.fromisoformat(day)
    vacant=32-len(data['rows'])
    if not vacant:return None
    active=min(16-len(data['active']),vacant)
    return dict(active=active,reserve=vacant-active,day=day)


def apply_batch(data,items,work):
    """One complete gallery transition, active slots first and total <=32."""
    na=min(work['active'],len(items));nb=min(work['reserve'],len(items)-na);items=items[:na+nb]
    if not items:return None
    hashes=[r['hash'] for r in items]
    if len(set(hashes))!=len(hashes) or set(hashes)&{r['hash'] for r in data['rows']}:raise ValueError('Duplicate batch photo')
    result=copy.deepcopy(data);result['active']+=hashes[:na];result['rows']+=items
    if len(result['rows'])>32 or len(result['active'])>16:raise ValueError('Photo batch exceeds vacant slots')
    result.update(revision=data['revision']+1,collection=dict(policy='fill-once-1',last_batch_day=work['day'],
        completed=len(result['rows'])==32,last_batch=dict(active=na,reserve=nb,total=na+nb)))
    return result


def administrator_change(data):
    previous=data.get('collection')
    if previous is not None:
        data['collection']=dict(previous,completed=len(data['rows'])==32)
