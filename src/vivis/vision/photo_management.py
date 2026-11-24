"""Atomic administrator edits shared by the local UI and CLI."""
import json
from vivis.vision.photo_pool import PhotoPool
from vivis.vision.photo_sampling import administrator_change

class RevisionConflict(ValueError):pass


def edit_active(path,person,activate=None,replace=None,deactivate=None,delete=None,lock_photo=None,unlock_photo=None,expected_revision=None):
    import fcntl,os
    if sum(bool(v) for v in (activate,deactivate,delete,lock_photo,unlock_photo))!=1:raise ValueError('请选择一种激活、停用、删除、锁定或解锁操作')
    if replace and (deactivate or delete or lock_photo or unlock_photo):raise ValueError('替换只能与激活同时使用')
    if replace and not activate:raise ValueError('替换时还需指定候选照片--activate')
    with (path.parent/'photo_pool.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        data=json.loads(path.read_text());PhotoPool.validate(data,person)
        if expected_revision is not None and data['revision']!=expected_revision:raise RevisionConflict('照片已变化，请刷新后重试')
        if unlock_photo and (unlock_photo not in data['active'] or any(r['hash']==unlock_photo and r['origin']=='manual' for r in data['rows'])):raise ValueError('只能解锁人工激活的自动照片，基础照片保持锁定')
        for row in data['rows']:
            resolved=(path.parent/row['path']).resolve()
            if not resolved.is_relative_to(path.parent.resolve()):raise ValueError('照片路径越界')
        hashes={r['hash'] for r in data['rows']}
        removed=deactivate or replace or delete
        protected={r['hash'] for r in data['rows'] if r['locked'] and r['origin']=='manual'}
        if removed in protected:raise ValueError('基础照片锁定，不能停用或替换；人工原图需单独重建登记')
        if removed and removed not in hashes:raise ValueError('照片已不存在，请刷新审核页')
        if (deactivate or replace) and removed not in data['active']:raise ValueError('要停用或替换的照片不在活动子集，请刷新审核页')
        if lock_photo and lock_photo not in data['active']:raise ValueError('只能锁定活动照片，请先激活候选')
        if activate and (activate not in hashes or activate in data['active']):
            raise ValueError('请选择候选照片的完整hash；已经活动的照片无需再次激活')
        if activate and not replace and len(data['active'])>=16:raise ValueError('激活照片已满16张，请先停用部分照片')
        if replace:data['active']=[activate if h==replace else h for h in data['active']]
        elif deactivate:data['active'].remove(deactivate)
        elif delete:data['active']=[h for h in data['active'] if h!=delete]
        elif activate:data['active'].append(activate)
        data['revision']+=1
        for row in data['rows']:
            if row['hash']==removed:row.update(excluded=True,locked=False)
            elif row['hash']==unlock_photo:row.update(locked=False,lock_reason='administrator_unlocked')
            elif row['hash'] in (activate,lock_photo):row.update(excluded=False,locked=True,lock_reason='administrator')
        deleted=next((r for r in data['rows'] if r['hash']==delete),None)
        if deleted:data['rows'].remove(deleted)
        administrator_change(data)
        PhotoPool.validate(data,person);tmp=path.with_suffix('.tmp')
        fd=os.open(tmp,os.O_CREAT|os.O_WRONLY|os.O_TRUNC,0o600)
        with os.fdopen(fd,'w') as f:
            json.dump(data,f,ensure_ascii=False);f.flush();os.fsync(f.fileno())
        tmp.replace(path)
        if deleted:(path.parent/deleted['path']).unlink(missing_ok=True)
        return data
