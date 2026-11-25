import copy
import unittest
from tools.evaluate_speaker import independent_cases


class HeldOutSpeakerTests(unittest.TestCase):
    def test_original_and_aggregate_training_sources_are_excluded_from_both_versions(self):
        cases=[dict(source_turn_id=t) for t in ['old','short1','short2','heldout']]
        gallery=dict(heyude=[dict(turn_id='old'),dict(turn_id='batch',source_turn_ids=['short1','short2'])])
        before=copy.deepcopy(gallery)
        self.assertEqual(independent_cases(cases,gallery),[dict(source_turn_id='heldout')])
        self.assertEqual(gallery,before)


if __name__=='__main__':unittest.main()
