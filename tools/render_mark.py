"""Render official Mark topology driven by exported A2F geometry in Blender."""
import bpy,json,sys,time,os
from pathlib import Path
import numpy as np
from mathutils import Vector,Matrix
root=Path(__file__).resolve().parent.parent
folder=root/'outputs/5_animation'
args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
mode=args[0] if args else 'preview'
parts=json.loads((folder/'mark_mesh.json').read_text())
frame_count=(folder/'skin.f32').stat().st_size//(61520*3*4)
skin=np.memmap(folder/'skin.f32',dtype='float32',mode='r',shape=(frame_count,61520,3))
tongue=np.memmap(folder/'tongue.f32',dtype='float32',mode='r',shape=(frame_count,5602,3))
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
    return np.column_stack((v[:,0],-v[:,2],v[:,1]-153))
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
            for face in eye.data.polygons:face.use_smooth=True

def aim(obj,target):obj.rotation_euler=(Vector(target)-obj.location).to_track_quat('-Z','Y').to_euler()
bpy.ops.object.camera_add(location=(0,-70,3))
camera=bpy.context.object;aim(camera,(0,0,0.5));camera.data.type='ORTHO';camera.data.ortho_scale=39
scene.camera=camera
for name,pos,power,size in [('Key',(-20,-30,25),16000,18),('Fill',(18,-20,8),9000,20),('Rim',(0,15,25),15000,15)]:
    data=bpy.data.lights.new(name,'AREA');data.energy=power;data.shape='DISK';data.size=size
    obj=bpy.data.objects.new(name,data);scene.collection.objects.link(obj);obj.location=pos;aim(obj,(0,0,0))
# Background plane behind the head.
bpy.ops.mesh.primitive_plane_add(size=200,location=(0,25,0),rotation=(np.pi/2,0,0))
bpy.context.object.data.materials.append(material('Background',(0.018,0.027,0.042)))
neutral_teeth=np.array(parts['lower_teeth']['vertices'],dtype=np.float32)
def update(frame):
    for name,data in [('skin',skin[frame]),('tongue',tongue[frame])]:
        objects[name].data.vertices.foreach_set('co',coords(data).reshape(-1));objects[name].data.update()
    transform=jaw[frame].reshape(4,4,order='F')
    vertices=neutral_teeth@transform[:3,:3].T+transform[:3,3]
    objects['lower_teeth'].data.vertices.foreach_set('co',coords(vertices).reshape(-1));objects['lower_teeth'].data.update()
output=root/'outputs/5_frames';output.mkdir(exist_ok=True)
indices=[0,180,480,900,1350] if mode=='preview' else list(range(0,frame_count,2))
start_index=int(os.environ.get('EVA_RENDER_START','0'))
end_index=int(os.environ.get('EVA_RENDER_END',str(len(indices))))
for index,frame in enumerate(indices):
    if mode!='preview' and not start_index<=index<end_index:continue
    update(frame)
    path=output/(f'preview_{frame:04d}.png' if mode=='preview' else f'{index:04d}.png')
    if mode!='preview' and path.exists():continue
    scene.render.filepath=str(path)
    start=time.time();bpy.ops.render.render(write_still=True)
    print('RENDERED',frame,path.name,round(time.time()-start,3),flush=True)
print('RENDER_COMPLETE',len(indices),flush=True)
