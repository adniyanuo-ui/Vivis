"""Loopback HTTP and disposable sample files; no devices or cloud clients."""
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from vivis.paths import ROOT
from vivis.interaction.review_server import ReviewServer


class ReviewServerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        for file in ('configs/interaction/continuity.json','configs/perception/familiar_faces.json','configs/perception/speaker_observation.json'):
            target=self.root/file;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/file).read_bytes())
        self.folder=self.root/'private/people/heyude';self.folder.mkdir(parents=True)
        (self.folder/'profile.json').write_text(json.dumps(dict(display_name='测试<script>')))
        self.hashes=[format(i,'064x') for i in range(18)]
        self.pool=dict(person_id='heyude',revision=1,active=self.hashes[:16],rows=[
            dict(hash=h,path='base.jpg' if i==0 else 'photo_pool/'+h+'.jpg',origin='manual' if i==0 else 'automatic',
                locked=i==0,vector=[1.]*128,coverage=[0.,0.,.5],quality=.9) for i,h in enumerate(self.hashes)])
        for row in self.pool['rows']:
            path=self.folder/row['path'];path.parent.mkdir(exist_ok=True);path.write_bytes(b'photo')
        self.poolpath=self.folder/'photo_pool.json';self.poolpath.write_text(json.dumps(self.pool))
        self.speaker=self.root/'private/speakers';(self.speaker/'heyude').mkdir(parents=True)
        self.c=json.loads((ROOT/'configs/perception/speaker_observation.json').read_text())
        gallery=dict(model_sha256=self.c['sha256'],people=dict(heyude=[dict(turn_id=str(i),vector=[1.,0.]) for i in range(3)]))
        self.gallery=self.speaker/'gallery.json';self.gallery.write_text(json.dumps(gallery))
        for i in range(3):(self.speaker/'heyude'/f'{i}.wav').write_bytes(b'RIFF0123456789')
        self.server=ReviewServer(self.root,0);self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.addCleanup(self.stop);self.port=self.server.server_port

    def stop(self):self.server.shutdown();self.server.server_close();self.thread.join(2)

    def request(self,path,method='GET',data=None,headers=None):
        conn=http.client.HTTPConnection('127.0.0.1',self.port,timeout=3);body=json.dumps(data).encode() if data is not None else None
        h={'Content-Type':'application/json'}
        if method=='POST':h.update(Origin=f'http://127.0.0.1:{self.port}',**{'X-Vivis-Review-Token':self.server.token})
        h.update(headers or {});conn.request(method,path,body,h);r=conn.getresponse();payload=r.read();status=r.status;response=dict(r.getheaders());conn.close()
        return status,response,payload

    def state(self):
        status,_,body=self.request('/api/state');self.assertEqual(status,200);return json.loads(body)

    def action(self,kind,sample,action,revision,**kw):
        return self.request('/api/action','POST',dict(kind=kind,person='heyude',id=sample,action=action,revision=revision,**kw))

    def test_state_is_read_only_and_does_not_expose_embeddings(self):
        before=self.gallery.read_bytes();data=self.state();p=data['people'][0]
        self.assertEqual(p['name'],'测试<script>');self.assertEqual(len(p['photos']),18);self.assertEqual(len(p['voices']),3)
        self.assertEqual(before,self.gallery.read_bytes());self.assertNotIn('vector',json.dumps(data))
        status,_,page=self.request('/');self.assertEqual(status,200);self.assertIn(b'textContent',page);self.assertNotIn(b'innerHTML',page)

    def test_photo_deactivation_activation_and_stale_revision(self):
        self.assertEqual(self.action('photo',self.hashes[1],'deactivate',1)[0],200)
        self.assertEqual(self.action('photo',self.hashes[16],'activate',1)[0],409)
        self.assertEqual(self.action('photo',self.hashes[16],'activate',2)[0],200)
        data=json.loads(self.poolpath.read_text());self.assertEqual(len(data['active']),16);self.assertTrue(data['rows'][16]['locked'])

    def test_full_subset_requires_explicit_atomic_replacement(self):
        before=self.poolpath.read_bytes();status,_,body=self.action('photo',self.hashes[16],'activate',1)
        self.assertEqual(status,400);self.assertEqual(json.loads(body)['error'],'激活照片已满16张，请先停用部分照片')
        self.assertEqual(before,self.poolpath.read_bytes())
        self.assertEqual(self.action('photo',self.hashes[16],'activate',1,replace=self.hashes[2])[0],200)
        self.assertEqual(json.loads(self.poolpath.read_text())['active'][2],self.hashes[16])

    def test_base_cannot_be_deleted_or_unlocked(self):
        for action in ('delete','deactivate','unlock'):self.assertEqual(self.action('photo',self.hashes[0],action,1)[0],400)
        self.assertTrue((self.folder/'base.jpg').exists())

    def test_voice_disable_activate_lock_unlock_delete(self):
        for action in ('deactivate','activate','unlock','lock','delete'):
            revision=self.state()['people'][0]['voice_revision'];self.assertEqual(self.action('voice','0',action,revision)[0],200)
        self.assertEqual(len(json.loads(self.gallery.read_text())['people']['heyude']),2)
        self.assertFalse((self.speaker/'heyude/0.wav').exists());self.assertTrue((self.speaker/'heyude/1.wav').exists())

    def test_cross_origin_and_wrong_token_or_host_cannot_mutate(self):
        request=dict(kind='voice',person='heyude',id='0',action='delete',revision=0);before=self.gallery.read_bytes()
        for headers in ({'Origin':'https://example.com'},{'X-Vivis-Review-Token':'wrong'},{'Host':'evil.example'}):
            self.assertEqual(self.request('/api/action','POST',request,headers)[0],403)
        self.assertEqual(before,self.gallery.read_bytes());self.assertEqual(self.request('/api/state',headers={'Host':'evil.example'})[0],403)

    def test_media_ranges_and_unregistered_paths(self):
        src=self.state()['people'][0]['voices'][0]['media'];status,headers,body=self.request(src,headers={'Range':'bytes=4-7'})
        self.assertEqual(status,206);self.assertEqual(body,b'0123');self.assertIn('Content-Range',headers)
        self.assertEqual(self.request(src,headers={'Range':'bytes=900-'})[0],416)
        self.assertEqual(self.request('/media/'+'f'*64)[0],404);self.assertEqual(self.request('/../../etc/passwd')[0],404)

    def test_symlink_outside_private_is_not_served(self):
        external=self.root/'secret.txt';external.write_text('secret');(self.folder/'base.jpg').unlink();(self.folder/'base.jpg').symlink_to(external)
        self.assertEqual(self.request('/api/state')[0],400)

    def test_arbitrary_actions_and_paths_rejected(self):
        self.assertEqual(self.action('voice','../gallery','delete',0)[0],400);self.assertEqual(self.action('voice','0','execute',0)[0],400)
        self.assertEqual(self.request('/api/action','POST',dict(kind='voice',person='heyude',id='0',action='delete',revision=0,path='/etc/passwd'))[0],400)


if __name__=='__main__':unittest.main()
