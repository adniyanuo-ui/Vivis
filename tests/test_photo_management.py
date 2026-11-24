"""Administrator changes keep bases, originals and the 16-photo boundary intact."""
import json
from pathlib import Path
import tempfile
import unittest

from tools.manage_photo_pool import edit_active,replace_active
from vivis.vision.photo_pool import choose


class PhotoManagementTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.folder=Path(self.tmp.name);self.path=self.folder/'photo_pool.json'
        self.hashes=[format(i,'064x') for i in range(20)]
        self.data=dict(person_id='heyude',revision=1,active=self.hashes[:16],rows=[
            dict(hash=h,path=('base.jpg' if i==0 else 'photo_pool/'+h+'.jpg'),locked=i==0,
                 origin='manual' if i==0 else 'automatic',vector=[1.]*128,quality=.9,coverage=[i*.1,0,.5])
            for i,h in enumerate(self.hashes)])
        self.path.write_text(json.dumps(self.data));(self.folder/'base.jpg').write_bytes(b'base')

    def test_deactivation_keeps_original_record_and_excludes_automatic_reselection(self):
        data=edit_active(self.path,'heyude',deactivate=self.hashes[2])
        self.assertEqual(len(data['active']),15);self.assertEqual(len(data['rows']),20)
        self.assertTrue(data['rows'][2]['excluded']);self.assertNotIn(self.hashes[2],choose(data['rows']))
        self.assertEqual((self.folder/'base.jpg').read_bytes(),b'base')

    def test_candidate_activation_fills_a_vacancy_and_can_restore_a_deactivated_photo(self):
        edit_active(self.path,'heyude',deactivate=self.hashes[2])
        data=edit_active(self.path,'heyude',activate=self.hashes[17])
        self.assertEqual(len(data['active']),16);self.assertIn(self.hashes[17],data['active'])
        self.assertTrue(data['rows'][17]['locked']);self.assertEqual(data['rows'][17]['lock_reason'],'administrator')
        edit_active(self.path,'heyude',deactivate=self.hashes[17])
        data=edit_active(self.path,'heyude',activate=self.hashes[2])
        self.assertIn(self.hashes[2],data['active']);self.assertFalse(data['rows'][2]['excluded'])

    def test_full_active_subset_cannot_grow_to_seventeen(self):
        before=self.path.read_bytes()
        with self.assertRaises(ValueError):edit_active(self.path,'heyude',activate=self.hashes[17])
        self.assertEqual(self.path.read_bytes(),before)

    def test_replacement_is_one_atomic_revision_and_excludes_the_removed_photo(self):
        data=replace_active(self.path,'heyude',self.hashes[17],self.hashes[2])
        self.assertEqual(data['revision'],2);self.assertEqual(len(data['active']),16)
        self.assertEqual(data['active'][2],self.hashes[17]);self.assertTrue(data['rows'][2]['excluded'])

    def test_base_and_stale_or_conflicting_actions_are_rejected_without_changing_manifest(self):
        for kw in (dict(deactivate=self.hashes[0]),dict(activate=self.hashes[17],replace=self.hashes[0]),
                   dict(deactivate='missing'),dict(activate=self.hashes[2]),dict(replace=self.hashes[2]),
                   dict(activate=self.hashes[17],deactivate=self.hashes[2])):
            before=self.path.read_bytes()
            with self.assertRaises(ValueError):edit_active(self.path,'heyude',**kw)
            self.assertEqual(self.path.read_bytes(),before)

    def test_deletion_updates_manifest_and_file_and_reopens_vacancies(self):
        photo=self.folder/self.data['rows'][17]['path'];photo.parent.mkdir();photo.write_bytes(b'original')
        self.data['collection']=dict(policy='fill-once-1',completed=False,last_batch_day='2026-11-23')
        self.path.write_text(json.dumps(self.data))
        data=edit_active(self.path,'heyude',delete=self.hashes[17])
        self.assertEqual(len(data['rows']),19);self.assertFalse(photo.exists())
        self.assertFalse(data['collection']['completed']);self.assertEqual(data['collection']['last_batch_day'],'2026-11-23')

    def test_existing_active_photo_can_be_manually_locked(self):
        data=edit_active(self.path,'heyude',lock_photo=self.hashes[2])
        self.assertTrue(data['rows'][2]['locked']);self.assertIn(self.hashes[2],choose(data['rows']))
        with self.assertRaises(ValueError):edit_active(self.path,'heyude',lock_photo=self.hashes[17])

    def test_delete_missing_or_supplied_base_does_not_change_manifest(self):
        for digest in ('missing',self.hashes[0]):
            before=self.path.read_bytes()
            with self.assertRaises(ValueError):edit_active(self.path,'heyude',delete=digest)
            self.assertEqual(self.path.read_bytes(),before)

    def test_explicit_unlock_keeps_active_and_stale_browser_cannot_overwrite(self):
        from vivis.vision.photo_management import RevisionConflict
        edit_active(self.path,'heyude',lock_photo=self.hashes[2],expected_revision=1)
        with self.assertRaises(RevisionConflict):edit_active(self.path,'heyude',deactivate=self.hashes[2],expected_revision=1)
        data=edit_active(self.path,'heyude',unlock_photo=self.hashes[2],expected_revision=2)
        self.assertFalse(data['rows'][2]['locked']);self.assertEqual(data['rows'][2]['lock_reason'],'administrator_unlocked')
        self.assertIn(self.hashes[2],data['active'])


if __name__=='__main__':unittest.main()
