"""Render official Mark topology driven by exported A2F geometry in Blender."""
import bpy,json,sys,time,os,wave,math
from pathlib import Path
import numpy as np
from mathutils import Vector,Matrix
root=Path(__file__).resolve().parent.parent

args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
folder=Path(args[0]).resolve()
mode=args[1] if len(args)>1 else 'preview'
parts=json.loads((folder/'mesh.json').read_text())
info=json.loads((folder/'export_info.json').read_text())
context=json.loads((folder/'metadata.json').read_text())
skin_count=info['skin_vertices'];tongue_count=info['tongue_vertices']
frame_count=(folder/'skin.f32').stat().st_size//(skin_count*3*4)
skin=np.memmap(folder/'skin.f32',dtype='float32',mode='r',shape=(frame_count,skin_count,3))
tongue=np.memmap(folder/'tongue.f32',dtype='float32',mode='r',shape=(frame_count,tongue_count,3))
jaw=np.memmap(folder/'jaw.f32',dtype='float32',mode='r',shape=(frame_count,16))
assert np.isfinite(skin).all() and np.isfinite(tongue).all() and np.isfinite(jaw).all()
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
scene=bpy.context.scene
scene.render.engine='CYCLES'
scene.cycles.samples=16
scene.cycles.use_denoising=True
scene.cycles.denoiser='OPENIMAGEDENOISE'
scene.cycles.denoising_use_gpu=True
scene.render.use_persistent_data=True
prefs=bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type='CUDA';prefs.get_devices()
gpus=[d for d in prefs.devices if d.type=='CUDA']
assert gpus,'No CUDA render device available'
chosen=gpus[-1]
for device in prefs.devices:device.use=(device.id==chosen.id)
scene.cycles.device='GPU'
print('RENDER_GPU',chosen.name,chosen.id,flush=True)
scene.render.resolution_x=768;scene.render.resolution_y=768;scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG'
scene.render.fps=30
scene.view_settings.view_transform='AgX'
scene.world.color=(0.035,0.035,0.035)
# Maya Y-up to Blender Z-up; face points toward negative Blender Y.
def coords(v):
    v=np.asarray(v,dtype=np.float32)
    return np.column_stack((v[:,0],-v[:,2],v[:,1]-142))
def material(name,color,roughness=0.55):
    m=bpy.data.materials.new(name);m.diffuse_color=(*color,1);m.use_nodes=True
    bsdf=m.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value=(*color,1)
    bsdf.inputs['Roughness'].default_value=roughness
    return m
skin_mat=material('Neutral clay',(0.48,0.57,0.65))
teeth_mat=material('Teeth',(0.78,0.77,0.69))
tongue_mat=material('Tongue',(0.33,0.085,0.08))
eye_mat=material('Sclera',(0.8,0.82,0.8),0.35)
iris_mat=material('Iris',(0.04,0.1,0.12),0.35)
pupil_mat=material('Pupil',(0.005,0.007,0.008),0.25)
objects={}
eye_details=[]
for name,data in parts.items():
    mesh=bpy.data.meshes.new(name)
    mesh.from_pydata(coords(data['vertices']).tolist(),[],data['faces']);mesh.update()
    obj=bpy.data.objects.new(name,mesh);scene.collection.objects.link(obj);objects[name]=obj
    obj.data.materials.append(skin_mat if name=='skin' else tongue_mat if name=='tongue' else eye_mat if 'eye' in name else teeth_mat)
    for face in mesh.polygons:face.use_smooth=True
    if 'eye' in name:
        original=np.array(data['vertices']);center=(original.min(axis=0)+original.max(axis=0))/2
        front=original[:,2].max()
        for label,radius,depth,mat in [('iris',0.57,0,iris_mat),('pupil',0.24,0.045,pupil_mat)]:
            pos=coords([[center[0],center[1],front+depth]])[0]
            bpy.ops.mesh.primitive_uv_sphere_add(segments=24,ring_count=12,radius=radius,location=pos)
            eye=bpy.context.object;eye.name=name+'_'+label;eye.scale.y=0.08;eye.data.materials.append(mat)
            eye_details.append((eye,Vector(coords([center])[0]),Vector(pos)))
            for face in eye.data.polygons:face.use_smooth=True

if context.get('face_appearance')=='warm_skin_visible_brows':
    # Paint on neutral mesh vertices: the color stays attached during every deformation.
    neutral=np.asarray(parts['skin']['vertices'],dtype=np.float32)
    x=np.abs(neutral[:,0]);u=np.clip((x-1.30)/3.65,0,1)
    center=149.05+.35*np.sin(np.pi*u)-.05*u
    half_width=.20+.08*np.sin(np.pi*u)-.12*u
    edge=np.clip((half_width+.09-np.abs(neutral[:,1]-center))/.13,0,1)
    extent=np.clip((x-1.20)/.22,0,1)*np.clip((5.08-x)/.26,0,1)
    front=np.clip((neutral[:,2]-2.1)/.6,0,1)
    alpha=edge*extent*front
    assert np.count_nonzero(alpha>.5)>40,'Eyebrow paint does not cover enough mesh vertices'
    color=np.array((.56,.34,.24))[None,:]*(1-alpha[:,None])+np.array((.025,.014,.009))[None,:]*alpha[:,None]
    rgba=np.column_stack((color,np.ones(len(color)))).astype(np.float32)
    attr=objects['skin'].data.color_attributes.new(name='FaceAppearance',type='FLOAT_COLOR',domain='POINT')
    attr.data.foreach_set('color',rgba.reshape(-1))
    nodes=skin_mat.node_tree.nodes
    paint=nodes.new('ShaderNodeVertexColor');paint.layer_name=attr.name
    shader=nodes.get('Principled BSDF')
    skin_mat.node_tree.links.new(paint.outputs['Color'],shader.inputs['Base Color'])
    shader.inputs['Roughness'].default_value=.52
    shader.inputs['Subsurface Weight'].default_value=.06
    print('EYEBROW_ATTACHED_VERTICES',int(np.count_nonzero(alpha>.5)),flush=True)

performance_root=None
if 'performance_pose' in context:
    # Optional artistic bust pose, separate from the A2F/FaceMotion animation.
    performance_root=bpy.data.objects.new('Performance pose',None)
    scene.collection.objects.link(performance_root)
    performance_root.matrix_world=Matrix.Translation((0,0,-8))
    for obj in list(scene.objects):
        if obj.type=='MESH':
            obj.parent=performance_root
            obj.matrix_parent_inverse=Matrix.Translation((0,0,8))

def aim(obj,target):obj.rotation_euler=(Vector(target)-obj.location).to_track_quat('-Z','Y').to_euler()
bpy.ops.object.camera_add(location=(0,-70,3))
camera=bpy.context.object;aim(camera,(0,0,0.5));camera.data.type='ORTHO';camera.data.ortho_scale=36.3
scene.camera=camera
for name,pos,power,size in [('Key',(-20,-30,25),16000,18),('Fill',(18,-20,8),9000,20),('Rim',(0,15,25),15000,15)]:
    data=bpy.data.lights.new(name,'AREA');data.energy=power;data.shape='DISK';data.size=size
    obj=bpy.data.objects.new(name,data);scene.collection.objects.link(obj);obj.location=pos;aim(obj,(0,0,0))
# Background plane behind the head.
bpy.ops.mesh.primitive_plane_add(size=200,location=(0,25,0),rotation=(np.pi/2,0,0))
bpy.context.object.data.materials.append(material('Background',(0.018,0.027,0.042)))
label='A2F v2.3.1 / Claire / Regression' if 'v2.3.1' in str(folder) else 'A2F v3.0 / Claire / Diffusion'
if 'FaceMotion52' in str(folder):label+=' / FaceMotion-52'
if 'EN_' in str(folder):label+=' / English'
if 'Welcome' in str(folder):label='Claire / Welcome Presentation'
if 'Expressive' in str(folder):label='A2F v3.0 / Claire / Expressive English' if 'EN_' in str(folder) else 'A2F v3.0 / Claire / Expressive Chinese'
if 'Singing' in str(folder):label='A2F v3.0 / Claire / Singing'
text_data=bpy.data.curves.new('Version label','FONT');text_data.body=label;text_data.align_x='CENTER';text_data.size=0.8
text_obj=bpy.data.objects.new('Version label',text_data);scene.collection.objects.link(text_obj)
text_obj.location=(0,-35,-15.0);text_obj.rotation_euler=(np.pi/2,0,0)
text_mat=bpy.data.materials.new('Label white');text_mat.use_nodes=True
nodes=text_mat.node_tree.nodes;nodes.clear();emission=nodes.new('ShaderNodeEmission');emission.inputs['Color'].default_value=(0.8,0.85,0.9,1)
out=nodes.new('ShaderNodeOutputMaterial');text_mat.node_tree.links.new(emission.outputs[0],out.inputs[0]);text_data.materials.append(text_mat)
neutral_teeth=np.array(parts['lower_teeth']['vertices'],dtype=np.float32)
def update(frame):
    if performance_root is not None:
        keys=context['performance_pose']['keyframes']
        t=timestamps[frame]
        angles=[np.interp(t,[k['time_seconds'] for k in keys],[k[n] for k in keys]) for n in ['pitch_deg','roll_deg','yaw_deg']]
        performance_root.rotation_euler=tuple(np.radians(angles))
    if 'performance_gaze' in context:
        keys=context['performance_gaze']['keyframes'];t=timestamps[frame]
        yaw=np.interp(t,[k['time_seconds'] for k in keys],[k['yaw_deg'] for k in keys])
        pitch=np.interp(t,[k['time_seconds'] for k in keys],[k['pitch_deg'] for k in keys])
        rotation=Matrix.Rotation(math.radians(yaw),3,'Z') @ Matrix.Rotation(math.radians(pitch),3,'X')
        for obj,center,position in eye_details:
            obj.location=center+rotation @ (position-center)
            obj.rotation_euler=rotation.to_euler()
    for name,data in [('skin',skin[frame]),('tongue',tongue[frame])]:
        objects[name].data.vertices.foreach_set('co',coords(data).reshape(-1));objects[name].data.update()
    transform=jaw[frame].reshape(4,4,order='F')
    vertices=neutral_teeth@transform[:3,:3].T+transform[:3,3]
    objects['lower_teeth'].data.vertices.foreach_set('co',coords(vertices).reshape(-1));objects['lower_teeth'].data.update()
    if blink_weights is not None:
        for side in ['left','right']:
            closed=blink_weights[side][frame]>0.94
            for suffix in ['iris','pupil']:
                bpy.data.objects[side+'_eye_'+suffix].hide_render=bool(closed)
    if 'emotion_schedule' in context and not context.get('hide_emotion_label',False):
        schedule=context['emotion_schedule'];keys=schedule['keyframes']
        weights=np.array([np.interp(timestamps[frame]*1000,[k['timestamp_ms'] for k in keys],[k['weights'][i] for k in keys]) for i in range(len(schedule['emotion_names']))])
        dominant=schedule['emotion_names'][int(weights.argmax())].capitalize() if weights.max()>0.05 else 'Neutral'
        text_data.body=label+' / '+dominant
output=folder/'frames';output.mkdir(exist_ok=True)
timestamps=np.loadtxt(folder/'timestamps.csv',delimiter=',')[:,1]/16000
context=json.loads((folder/'metadata.json').read_text())
blink_weights=None
if 'blink_schedule' in context:
    protocol_frames=[json.loads(line) for line in Path(context['interface_source']).read_text().splitlines()]
    assert len(protocol_frames)==frame_count
    blink_weights={side:np.array([frame['blendshapes']['eyeBlink'+side.capitalize()] for frame in protocol_frames]) for side in ['left','right']}
with wave.open(context.get('audio',str(root/'5.wav'))) as audio:
    duration=audio.getnframes()/audio.getframerate()
video_frame_count=math.ceil(duration*30)
video_times=np.arange(video_frame_count)/30
source_indices=np.abs(timestamps[:,None]-video_times[None,:]).argmin(axis=0)
assert np.max(np.abs(timestamps[source_indices]-video_times))<0.001
preview_frames=[0,90,240,450,675]
if 'blink_schedule' in context:
    peak=context['blink_schedule']['events'][0]['peak_seconds']
    preview_frames=[0,round((peak-0.12)*30),round(peak*30),round((peak+0.1)*30),round((peak+0.25)*30)]
if 'preview_times_seconds' in context:
    preview_frames=[round(t*30) for t in context['preview_times_seconds']]
indices=[min(i,video_frame_count-1) for i in preview_frames] if mode=='preview' else list(range(video_frame_count))
start_index=int(os.environ.get('EVA_RENDER_START','0'))
end_index=int(os.environ.get('EVA_RENDER_END',str(len(indices))))
for index,frame in enumerate(indices):
    if mode!='preview' and not start_index<=index<end_index:continue
    update(int(source_indices[frame]))
    path=output/(f'preview_{frame:04d}.png' if mode=='preview' else f'{index:04d}.png')
    if mode!='preview' and path.exists():continue
    scene.render.filepath=str(path)
    start=time.time();bpy.ops.render.render(write_still=True)
    print('RENDERED',frame,path.name,round(time.time()-start,3),flush=True)
print('RENDER_COMPLETE',len(indices),flush=True)
