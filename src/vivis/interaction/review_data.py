"""Local review data and finite actions; no devices, models or cloud clients."""
import hashlib
import json
from pathlib import Path
import sqlite3

from vivis.interaction.memory import person_key
from vivis.speech.speaker_gallery import read_gallery,edit_reference,sample_key
from vivis.vision.photo_management import edit_active
from vivis.vision.photo_pool import PhotoPool


class ReviewData:
    def __init__(self,root):
        self.root=Path(root).resolve();self.private=self.root/'private'
        continuity=json.loads((self.root/'configs/interaction/continuity.json').read_text())
        familiar=json.loads((self.root/'configs/perception/familiar_faces.json').read_text())
        self.speaker=json.loads((self.root/continuity['speaker_config']).read_text())
        self.people=self.private_path(self.root/familiar['people_directory'])
        self.archive=self.private_path(self.root/continuity['directory'])
        self.speakers=self.private_path(self.root/self.speaker['directory'])

    def private_path(self,path):
        path=Path(path).resolve()
        if not path.is_relative_to(self.private.resolve()):raise ValueError('私有数据路径越界')
        return path

    def person_folder(self,person):
        person_key(person);folder=self.private_path(self.people/person)
        if not folder.is_relative_to(self.people) or not (folder/'profile.json').is_file():raise ValueError('人物不存在')
        return folder

    def photo_pool(self,person):
        path=self.person_folder(person)/'photo_pool.json'
        data=json.loads(path.read_text());PhotoPool.validate(data,person)
        return path,data

    def speaker_gallery(self):return read_gallery(self.speakers/'gallery.json',self.speaker['sha256'])

    def media_id(self,path):
        path=self.private_path(path)
        return hashlib.sha256(str(path.relative_to(self.private)).encode()).hexdigest()

    def snapshot(self):
        gallery=self.speaker_gallery();people=[];media={}
        def url(path,base):
            path=self.private_path(path)
            if not path.is_relative_to(base) or not path.is_file():return None
            key=self.media_id(path);media[key]=path
            return '/media/'+key
        dbpath=self.archive/'rounds.sqlite3';db=None
        if dbpath.exists():
            db=sqlite3.connect(self.private_path(dbpath).as_uri()+'?mode=ro',uri=True,timeout=1)
            db.row_factory=sqlite3.Row
        try:
            for profile in sorted(self.people.glob('*/profile.json')):
                pid=profile.parent.name;folder=self.person_folder(pid)
                person=dict(id=pid,name=json.loads(profile.read_text()).get('display_name') or pid,photos=[],voices=[],rounds=[])
                pool_path=folder/'photo_pool.json'
                if pool_path.exists():
                    _,pool=self.photo_pool(pid);person['photo_revision']=pool['revision']
                    for row in pool['rows']:
                        person['photos'].append(dict(id=row['hash'],active=row['hash'] in pool['active'],locked=row['locked'],
                            base=row['origin']=='manual',origin=row['origin'],excluded=row.get('excluded',False),
                            media=url(folder/row['path'],folder),filename=Path(row['path']).name))
                person['voice_revision']=gallery['revision']
                for row in gallery['people'].get(pid,[]):
                    person['voices'].append(dict(id=row['turn_id'],active=row.get('active',True),locked=row.get('locked',False),
                        created=row.get('created'),media=url(self.speakers/pid/(row['turn_id']+'.wav'),self.speakers)))
                if db:
                    for item in db.execute('SELECT * FROM rounds WHERE person=? ORDER BY created DESC LIMIT 10',(pid,)):
                        r=dict(id=item['id'],created=item['created'],user_text=item['user_text'],outcome=item['outcome'],replies=[])
                        r['media']=url(self.archive/item['audio'],self.archive) if item['audio'] else None
                        for reply in db.execute('SELECT text,audio,outcome FROM replies WHERE round_id=?',(item['id'],)):
                            r['replies'].append(dict(text=reply['text'],outcome=reply['outcome'],
                                media=url(self.archive/reply['audio'],self.archive) if reply['audio'] else None))
                        person['rounds'].append(r)
                people.append(person)
        finally:
            if db:db.close()
        return dict(people=people),media

    def apply(self,request):
        if not isinstance(request,dict) or set(request)-{'kind','person','id','action','revision','replace'}:raise ValueError('请求字段不正确')
        kind=request.get('kind');person=request.get('person');sample=request.get('id');action=request.get('action');revision=request.get('revision')
        if not isinstance(person,str) or type(revision) is not int or revision<0:raise ValueError('人物或版本不正确，请刷新')
        folder=self.person_folder(person)
        if kind=='photo':
            if not isinstance(sample,str) or len(sample)!=64 or any(c not in '0123456789abcdef' for c in sample):raise ValueError('照片编号不正确')
            names=dict(activate='activate',deactivate='deactivate',delete='delete',lock='lock_photo',unlock='unlock_photo')
            if action not in names:raise ValueError('无效照片操作')
            kwargs={names[action]:sample,'expected_revision':revision}
            if request.get('replace'):
                if action!='activate' or not isinstance(request['replace'],str):raise ValueError('替换只能用于激活')
                kwargs['replace']=request['replace']
            edit_active(folder/'photo_pool.json',person,**kwargs)
        elif kind=='voice':
            sample_key(sample)
            if request.get('replace'):raise ValueError('声纹样本不做自动替换')
            edit_reference(self.speakers/'gallery.json',self.speaker['sha256'],person,sample,action,revision)
        else:raise ValueError('无效样本类型')
        return dict(ok=True,message='已保存；识别程序在后续处理时读取新样本清单。')
