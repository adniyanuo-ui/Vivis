"""Onsite language examples and head-fixed roll / warm-transition regressions."""
import unittest
import numpy as np
from pathlib import Path
from vivis.paths import ROOT
from vivis.motion.mapping import read_json
from vivis.runtime.robot import Robot
from vivis.runtime.homing import smooth_home_targets
from vivis.interaction.dialogue import program_command
from vivis.interaction.intents import IntentRouter
from vivis.vision.independent_follow import IndependentFaceFollower


class UserFeedbackTests(unittest.TestCase):
    def test_observed_queries_and_commands_keep_wake_and_negation_guards(self):
        r=IntentRouter(read_json(ROOT/'configs/interaction/programs.json'))
        for text in ('干点啥','帮我干点啥','你们帮我干点啥','你有啥功能','你会做什么','嗯有什么功能'):
            self.assertEqual(r.parse(text,state='interaction'),dict(kind='prompt',name='menu'),text)
            self.assertEqual(r.parse(text,state='standby')['kind'],'ignore')
        self.assertEqual(r.parse('你是谁'),dict(kind='prompt',name='identity'))
        for text in ('播放魏什么剧版','播放魏总剧本'):
            self.assertEqual(r.parse(text),dict(kind='select',program='wz2'))
        for text in ('模仿我的表情','模仿表情'):
            self.assertEqual(r.parse(text),dict(kind='select',program='expression-follow'))
        for text in ('不要播放魏什么剧版','他说模仿我的表情','嗯不要停止','帮我干点啥是什么意思'):
            self.assertEqual(r.parse(text)['kind'],'ignore',text)

    def test_home_cap_reduces_planned_duration_without_changing_endpoint_shape(self):
        start={32:1655.,20:2245.};neutral={32:2047.,20:2047.}
        cold,a=smooth_home_targets(start,neutral,60);warm,b=smooth_home_targets(start,neutral,120)
        self.assertEqual(cold,12.25);self.assertEqual(warm,6.125)
        for u in (0.,.1,.5,.9,1.):self.assertEqual(a(cold*u),b(warm*u))
        robot=Robot(ROOT/'configs/robots/xiaoshuai/robot.json')
        settings=read_json(ROOT/'configs/interaction/p1.json');speech=read_json(ROOT/settings['speech'])
        for program in (None,dict(kind='interaction'),dict(kind='prerecorded',work='0001'),dict(kind='expression_follow')):
            cold_command=program_command(program,robot,Path('/tmp/cold'),speech,settings)
            warm_command=program_command(program,robot,Path('/tmp/warm'),speech,{**settings,'_home_speed_counts_s':120})
            self.assertNotIn('--home-speed',cold_command)
            self.assertEqual(warm_command[warm_command.index('--home-speed')+1],'120')

    def simulate_roll(self,mode,sign=1):
        robot=Robot(ROOT/'configs/robots/xiaoshuai/robot.json');config=robot.tracking()
        config['independent_neck'].update(roll_control_mode=mode)
        config['independent_neck']['roll']['sign']=sign
        follower=IndependentFaceFollower(config,robot.motion_profile())
        eye=follower.eye_id;goals={eye:2047.,33:2047.,34:2047.};follower.reset(goals,0.)
        values=[]
        # Synthetic plant:200 counts/radian, exact encoder response. This is a
        # software closed-loop regression, not an identified hardware model.
        for i in range(1,601):
            now=i/60;user_roll=.3 if now<5. else 0.
            robot_roll=sign*(goals[34]-2047)/200.
            packet=dict(timestamp_monotonic=now,target=dict(x=0.,y=0.,eye_line_roll_rad=user_roll-robot_roll))
            for sid,value in goals.items():follower.observe(sid,value,now)
            goals.update(follower.step(packet,'tracking',1/60,now))
            values.append(sign*(goals[34]-2047)/200.)
            motor=robot.axes[34]
            self.assertTrue(motor['device_min']<=goals[34]<=motor['device_max'])
        return follower,goals,np.array(values)

    def test_residual_roll_servo_follows_then_recovers_instead_of_restoring_toward_neutral_early(self):
        _,_,old=self.simulate_roll('position')
        for sign in (1,-1):
            follower,goals,new=self.simulate_roll('visual_velocity',sign)
            self.assertLess(old[290],.14);self.assertGreater(new[290],.27)
            self.assertLess(abs(new[-1]),.01)
            for sid,value in goals.items():follower.observe(sid,value,10.1)
            follower.step(dict(timestamp_monotonic=10.1,target=dict(x=0.,y=0.,eye_line_roll_rad=None)),
                          'tracking',1/60,10.1)
            self.assertEqual(follower.diagnostic['requested_velocity'][34],0.)


if __name__=='__main__':unittest.main()
