"""Replay the ID29 startup2055 ->2052 premature-arrival failure."""
import itertools,unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
from vivis.runtime.robot import Robot
from vivis.runtime.show import recover_tongue,tongue_at_startup_goal,prepare_recovery_clearance,recovery_clearance_plan


class TongueArrivalTests(unittest.TestCase):
    def fixture(self,samples):
        robot=Robot();motors={m['id']:m for m in robot.motion_profile()['motors']}
        motor=motors[29];motor['initial_settle_tolerance_counts']=40
        def state(position):
            return dict(position=position,load_raw=64,voltage_raw=63,temperature_c=35,status_register=0)
        snapshot=dict(id=29,model_raw=521,mode=0,torque_enable=0,configured_min=1400,configured_max=2050,
                      position_raw=2055,**{k:v for k,v in state(2055).items() if k!='position'})
        bus=Mock();bus.snapshot.return_value=snapshot;enabled=[False]
        bus.read.side_effect=lambda sid,a,n:bytes([enabled[0]])
        def enable(*args):enabled[0]=True;return 0,0
        bus.sdk.write1ByteTxRx.side_effect=enable
        values=itertools.chain(samples,itertools.repeat(samples[-1]))
        bus.feedback.side_effect=lambda sid:state(next(values))
        now=[1.];clock=SimpleNamespace(monotonic=lambda:now[0],sleep=lambda duration:now.__setitem__(0,now[0]+duration))
        return robot,motors,motor,snapshot,bus,clock

    def test_neutral_tolerance_does_not_hide_feedback_range_violation(self):
        _,_,motor,*_=self.fixture([2047])
        for position in (2052,2055,2087,1400):self.assertFalse(tongue_at_startup_goal(position,motor))
        for position in (2047,2049,2050):self.assertTrue(tongue_at_startup_goal(position,motor))
        self.assertFalse(tongue_at_startup_goal(2052,{**motor,'feedback_max':2087}))

    def test_near_neutral_but_outside_range_still_opens_jaw_before_recovery(self):
        robot,motors,motor,snapshot,bus,clock=self.fixture([2047])
        plan=recovery_clearance_plan(robot.home(),motors);report={}
        with patch('vivis.runtime.show.home_all') as home:
            prepare_recovery_clearance(bus,plan,[snapshot],motors,lambda:None,report,60)
        self.assertTrue(report['required']);self.assertEqual(home.call_args.args[1][0]['neutral'],1900)
        self.assertEqual(home.call_args.args[5],60)

    def test_captured2055_2052_samples_are_not_counted_as_arrived(self):
        _,_,motor,_,bus,clock=self.fixture([2055,2052,2049,2048]);events=[]
        with patch('vivis.runtime.show.time',clock):recover_tongue(bus,motor,lambda:None,events,speed=60)
        self.assertEqual(bus.feedback.call_count,4)
        self.assertEqual([s['position'] for s in events if 'position' in s],[2055,2052,2049,2048])
        self.assertEqual([s['startup_goal_ready'] for s in events if 'position' in s],[False,False,True,True])
        for call in bus.goals.call_args_list:self.assertEqual(call.args,({29:2047},{29:60}))
        self.assertEqual(motor['feedback_max'],2050)

    def test_persistent_outside_range_times_out_without_expanding_limits(self):
        _,_,motor,_,bus,clock=self.fixture([2052])
        with patch('vivis.runtime.show.time',clock),self.assertRaisesRegex(RuntimeError,'did not reach its authorized startup corridor'):
            recover_tongue(bus,motor,lambda:None,[],speed=60)
        self.assertGreater(bus.feedback.call_count,2);self.assertEqual(motor['feedback_max'],2050)
        self.assertEqual(bus.goals.call_count,2)
