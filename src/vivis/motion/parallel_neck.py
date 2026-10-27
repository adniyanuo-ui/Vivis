"""Two rotary crank/rod limbs with a central universal-joint pivot.

Geometry is measured input, never inferred from a motor's ID or register limits.
Axes: x right, y forward, z up; R=Rx(pitch)@Ry(roll), angles in radians.
Fixed rod lengths constrain both joint commands together. No actuator I/O.
"""
import math
import numpy as np


class UnreachablePose(ValueError):
    pass


def _number(value,name):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError('Invalid parallel neck '+name)
    return float(value)


def _vector(value,name):
    if not isinstance(value,list) or len(value)!=3:
        raise ValueError('Need three measured coordinates: '+name)
    return np.array([_number(v,name) for v in value])


def rotation(pitch,roll):
    cp,sp,cr,sr=math.cos(pitch),math.sin(pitch),math.cos(roll),math.sin(roll)
    return np.array([[cr,0,sr],[sp*sr,cp,-sp*cr],[-cp*sr,sp,cp*cr]])


def rotation_derivatives(pitch,roll):
    cp,sp,cr,sr=math.cos(pitch),math.sin(pitch),math.cos(roll),math.sin(roll)
    return (np.array([[0,0,0],[cp*sr,-sp,-cp*cr],[sp*sr,cp,-sp*cr]]),
            np.array([[-sr,0,cr],[sp*cr,0,sp*sr],[-cp*cr,0,-cp*sr]]))


class RotaryParallelNeck:
    def __init__(self,geometry):
        if geometry.get('schema_version')!='vivis-rotary-parallel-neck-1.0':
            raise ValueError('Unsupported parallel neck geometry')
        self.pivot=_vector(geometry['pivot_base_mm'],'pivot_base_mm')
        limits=geometry['pose_limits_rad']
        if not isinstance(limits,list) or len(limits)!=2 or any(not isinstance(row,list) or len(row)!=2 for row in limits):
            raise ValueError('Need two pitch/roll intervals')
        self.pose_limits=np.array([[_number(v,'pose limit') for v in row] for row in limits])
        if (self.pose_limits.shape!=(2,2) or not np.isfinite(self.pose_limits).all()
                or not np.all(self.pose_limits[:,0]<0) or not np.all(self.pose_limits[:,1]>0)
                or np.max(np.abs(self.pose_limits))>=math.pi/2):
            raise ValueError('Need bounded pitch/roll intervals around neutral')
        self.condition_limit=_number(geometry.get('jacobian_condition_limit',50.),'condition limit')
        if self.condition_limit<1:raise ValueError('Invalid parallel Jacobian condition limit')
        self.branches=[]
        for item in geometry['branches']:
            sid=item['id']
            if type(sid) is not int or sid not in (33,34):raise ValueError('Parallel neck axes must be33/34')
            u=_vector(item['crank_u'],'crank_u');v=_vector(item['crank_v'],'crank_v')
            if abs(np.linalg.norm(u)-1)>1e-6 or abs(np.linalg.norm(v)-1)>1e-6 or abs(u@v)>1e-6:
                raise ValueError('Crank plane must have two orthonormal measured basis vectors')
            branch=dict(id=sid,pivot=_vector(item['motor_pivot_mm'],'motor_pivot_mm'),u=u,v=v,
                anchor=_vector(item['platform_anchor_mm'],'platform_anchor_mm'),
                radius=_number(item['crank_radius_mm'],'crank radius'),
                length=_number(item['rod_length_mm'],'rod length'),
                q0=_number(item['neutral_crank_angle_rad'],'neutral crank angle'),
                sign=item['encoder_sign'],neutral=item.get('neutral_count',2047),
                low=_number(item['minimum_count'],'minimum count'),high=_number(item['maximum_count'],'maximum count'),
                scale=_number(item.get('counts_per_revolution',4096),'encoder scale')/(2*math.pi))
            if (type(branch['sign']) is not int or branch['sign'] not in (-1,1)
                    or branch['neutral']!=2047 or branch['radius']<=0 or branch['length']<=0
                    or branch['scale']<=0 or not 0<=branch['low']<2047<branch['high']<=4095):
                raise ValueError('Invalid parallel branch parameters')
            w=self.pivot+branch['anchor']-branch['pivot']
            q0=branch['q0'];crank=branch['radius']*(u*math.cos(q0)+v*math.sin(q0))
            if abs(np.linalg.norm(w-crank)-branch['length'])>.01:
                raise ValueError('Measured neutral geometry does not close the fixed rod')
            phase=math.atan2(w@v,w@u)
            assembly=math.sin(q0-phase)
            if abs(assembly)<1e-5:raise ValueError('Neutral assembly is at a crank toggle')
            branch['assembly']=1 if assembly>0 else -1
            self.branches.append(branch)
        if [b['id'] for b in self.branches]!=[33,34]:raise ValueError('Need ordered branches33/34')
        self.inverse(0.,0.)

    def _solve(self,pitch,roll,check_joint_limits=True):
        pose=np.array([_number(pitch,'pitch'),_number(roll,'roll')])
        if np.any(pose<self.pose_limits[:,0]) or np.any(pose>self.pose_limits[:,1]):
            raise UnreachablePose('Pitch/roll outside measured pose envelope')
        matrix=rotation(*pose);angles=[];counts={}
        for b in self.branches:
            w=self.pivot+matrix@b['anchor']-b['pivot']
            a,c=float(w@b['u']),float(w@b['v']);amplitude=math.hypot(a,c)
            if amplitude<1e-9:raise UnreachablePose('Crank inverse is degenerate')
            ratio=(w@w+b['radius']**2-b['length']**2)/(2*b['radius']*amplitude)
            if not -1+1e-10<ratio<1-1e-10:raise UnreachablePose('Fixed rod cannot reach pose or is at toggle')
            angle=math.atan2(c,a)+b['assembly']*math.acos(ratio)
            angle=b['q0']+(angle-b['q0']+math.pi)%(2*math.pi)-math.pi
            count=b['neutral']+b['sign']*b['scale']*(angle-b['q0'])
            if check_joint_limits and not b['low']-1e-7<=count<=b['high']+1e-7:
                raise UnreachablePose('Coupled command exceeds ID'+str(b['id'])+' envelope')
            counts[b['id']]=float(count);angles.append(angle)
        return pose,angles,counts

    def inverse(self,pitch,roll):
        pose,angles,counts=self._solve(pitch,roll)
        self._jacobian(pose,angles)
        return counts

    def _jacobian(self,pose,angles):
        matrix=rotation(*pose);derivatives=rotation_derivatives(*pose);rows=[]
        for b,angle in zip(self.branches,angles):
            w=self.pivot+matrix@b['anchor']-b['pivot']
            rod=w-b['radius']*(b['u']*math.cos(angle)+b['v']*math.sin(angle))
            tangent=b['radius']*(-b['u']*math.sin(angle)+b['v']*math.cos(angle))
            denominator=float(rod@tangent)
            if abs(denominator)<1e-6*b['length']*b['radius']:
                raise UnreachablePose('Parallel crank toggle approaches singularity')
            rows.append([b['sign']*b['scale']*float(rod@(d@b['anchor']))/denominator for d in derivatives])
        jacobian=np.array(rows)
        if not np.isfinite(jacobian).all() or np.linalg.cond(jacobian)>self.condition_limit:
            raise UnreachablePose('Parallel pitch/roll Jacobian is singular or ill-conditioned')
        return jacobian

    def jacobian(self,pitch,roll):
        pose,angles,_=self._solve(pitch,roll)
        return self._jacobian(pose,angles)

    def forward(self,counts,initial=(0.,0.),measurement_tolerance_counts=0.):
        tolerance=_number(measurement_tolerance_counts,'measurement tolerance')
        if tolerance<0:raise ValueError('Negative parallel measurement tolerance')
        tips=[]
        for b in self.branches:
            count=_number(counts[b['id']],'encoder count')
            if not b['low']-tolerance<=count<=b['high']+tolerance:
                raise UnreachablePose('Measured joints outside reviewed envelope')
            angle=b['q0']+b['sign']*(count-b['neutral'])/b['scale']
            tips.append(b['pivot']+b['radius']*(b['u']*math.cos(angle)+b['v']*math.sin(angle)))
        pose=np.array([_number(v,'initial pose') for v in initial])
        if pose.shape!=(2,):raise ValueError('Need two initial platform angles')
        pose=np.clip(pose,self.pose_limits[:,0],self.pose_limits[:,1])
        for _ in range(20):
            matrix=rotation(*pose)
            rods=[self.pivot+matrix@b['anchor']-tip for b,tip in zip(self.branches,tips)]
            lengths=[float(np.linalg.norm(rod)) for rod in rods]
            error=np.array([length-b['length'] for length,b in zip(lengths,self.branches)])
            if np.max(np.abs(error))<1e-7:break
            derivatives=rotation_derivatives(*pose)
            jac=np.array([[float(rod@(d@b['anchor']))/length for d in derivatives]
                for b,rod,length in zip(self.branches,rods,lengths)])
            if np.linalg.cond(jac)>self.condition_limit:raise UnreachablePose('Forward Jacobian is ill-conditioned')
            delta=np.linalg.solve(jac,error)
            delta*=min(1.,.2/max(1e-12,float(np.max(np.abs(delta)))))
            pose=np.clip(pose-delta,self.pose_limits[:,0],self.pose_limits[:,1])
        else:
            raise UnreachablePose('Parallel forward solve did not close both rods')
        _,angles,check=self._solve(*map(float,pose),check_joint_limits=False)
        self._jacobian(pose,angles)
        if any(abs(check[b['id']]-counts[b['id']])>1e-3 for b in self.branches):
            raise UnreachablePose('Forward result belongs to another assembly branch')
        return tuple(map(float,pose))
