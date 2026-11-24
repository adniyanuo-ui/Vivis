"""Private review export does not mutate journals or trust embedded markup/paths."""
import json
from pathlib import Path
import tempfile
import unittest

from tools.review_continuity import export
from vivis.paths import ROOT
from vivis.interaction.continuity import RoundStore


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        for file in ('configs/interaction/continuity.json','configs/perception/familiar_faces.json','configs/perception/speaker_observation.json'):
            target=self.root/file;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/file).read_bytes())
        folder=self.root/'private/people/heyude';folder.mkdir(parents=True)
        (folder/'profile.json').write_text(json.dumps(dict(display_name='测试<script>alert(1)</script>')))
        (folder/'photo.png').write_bytes(b'image')
        self.pool=dict(rows=[dict(hash='abc',path='photo.png',locked=True,origin='manual')],active=['abc'])
        (folder/'photo_pool.json').write_text(json.dumps(self.pool))
        config=json.loads((ROOT/'configs/interaction/continuity.json').read_text());self.store=RoundStore(config,self.root)
        rid,_=self.store.begin('test','user','heyude');self.store.final(rid,'我喜欢苹果 <script>alert(2)</script>')
        self.store.close(rid,'complete');self.store.reply('reply','test',rid,text='您好',outcome='interrupted')
        self.output=self.root/'private/review/index.html'

    def test_export_contains_photo_status_and_original_text_but_no_script_execution(self):
        export(self.root,self.output);text=self.output.read_text()
        self.assertIn('基础照 · 已锁定',text);self.assertIn('已被打断',text)
        self.assertIn('我喜欢苹果 &lt;script&gt;',text);self.assertNotIn('<script>',text)
        self.assertIn('../people/heyude/photo.png',text);self.assertIn('声纹参考 0 段',text)

    def test_export_keeps_original_database_and_photos_unchanged(self):
        before=self.store.path.read_bytes();photo=self.root/'private/people/heyude/photo.png'
        export(self.root,self.output,person='heyude')
        self.assertEqual(self.store.path.read_bytes(),before);self.assertEqual(photo.read_bytes(),b'image')

    def test_output_must_remain_private(self):
        with self.assertRaises(ValueError):export(self.root,self.root/'public.html')
        self.assertFalse((self.root/'public.html').exists())

    def test_database_cannot_be_selected_as_output(self):
        before=self.store.path.read_bytes()
        with self.assertRaises(ValueError):export(self.root,self.store.path)
        self.assertEqual(self.store.path.read_bytes(),before)

    def test_arbitrary_local_files_cannot_be_embedded_by_pool_path(self):
        self.pool['rows'][0]['path']='../../../secret.txt'
        (self.root/'secret.txt').write_text('secret')
        (self.root/'private/people/heyude/photo_pool.json').write_text(json.dumps(self.pool))
        export(self.root,self.output);text=self.output.read_text()
        self.assertNotIn('src="../../../secret.txt"',text);self.assertIn('原图不可用',text)

    def test_commands_are_only_text_and_locked_bases_have_no_deactivate_command(self):
        export(self.root,self.output);text=self.output.read_text()
        self.assertNotIn('--deactivate',text)
        self.pool['rows'].append(dict(hash='candidate',path='candidate.jpg',locked=False,origin='automatic',excluded=True))
        (self.root/'private/people/heyude/photo_pool.json').write_text(json.dumps(self.pool))
        export(self.root,self.output);text=self.output.read_text()
        self.assertIn('--activate candidate',text);self.assertIn('已停用',text)
        self.assertNotIn('<script>',text);self.assertNotIn('<form',text)


if __name__=='__main__':unittest.main()
