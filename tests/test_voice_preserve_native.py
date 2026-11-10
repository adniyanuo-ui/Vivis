"""Optional compiled extension checks; never open hardware or use private recordings."""
import importlib.util
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('_vivis_aec3'),'optional compiled speech environment only')
class NativeProfileTests(unittest.TestCase):
    def test_constructor_guards_and_original_module_can_coexist(self):
        import _vivis_aec3
        from pywebrtc_audio import AudioProcessor as Original
        self.assertEqual(_vivis_aec3.vivis_profile_api_version,1)
        for threshold in (.24,2.01,float('nan'),float('inf')):
            with self.assertRaises(ValueError):_vivis_aec3.AudioProcessor(nearend_threshold=threshold)
        near=np.sin(np.arange(320)*.12).astype(np.float32)*.05;far=np.zeros(320,np.float32)
        for factory in (Original,_vivis_aec3.AudioProcessor):
            engine=factory(echo_cancellation=True,noise_suppression=True,auto_gain_control=False)
            result=engine.process(near,far)
            self.assertEqual(result.shape,(320,));self.assertTrue(np.isfinite(result).all())

    def test_reset_retains_the_candidate_constructor_setting(self):
        import _vivis_aec3
        rng=np.random.default_rng(137);far=rng.normal(0,.08,320*180).astype(np.float32)
        t=np.arange(len(far))/16000
        near=(np.r_[np.zeros(1536),far[:-1536]]*.4+.08*np.sin(2*np.pi*173*t)*np.sin(2*np.pi*3*t)**2).astype(np.float32)
        options=dict(echo_cancellation=True,noise_suppression=True,nearend_threshold=1.,auto_gain_control=False)
        used=_vivis_aec3.AudioProcessor(**options)
        def run(engine):
            return np.concatenate([engine.process(near[i:i+320].copy(),far[i:i+320].copy()) for i in range(0,len(near),320)])
        run(used);used.reset();restored=run(used)
        fresh=run(_vivis_aec3.AudioProcessor(**options))
        np.testing.assert_allclose(restored,fresh,atol=2e-6,rtol=1e-5)


if __name__=='__main__':unittest.main()
