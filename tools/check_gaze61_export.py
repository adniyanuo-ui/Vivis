"""Independent Blender/mathutils verification against actual preview transforms."""
import json,sys,math
from pathlib import Path
import numpy as np
from mathutils import Euler,Matrix

args=sys.argv[sys.argv.index('--')+1:]
folder=Path(args[0]);report_path=Path(args[1])
meta=json.loads((folder/'metadata.json').read_text())
report=json.loads(report_path.read_text())
times=np.loadtxt(folder/'timestamps.csv',delimiter=',')[:,1]/16000

def angles(field,names,second):
    keys=meta[field]['keyframes']
    return [float(np.interp(second,[k['time_seconds'] for k in keys],[k[n] for k in keys])) for n in names]

def restore(y,p,r):
    # Map face-local Y(up), X(subject-left), Z(forward) to Blender Z, X, -Y.
    return Matrix.Rotation(y,3,'Z') @ Matrix.Rotation(p,3,'X') @ Matrix.Rotation(-r,3,'Y')

checks=[]
for output in report['outputs']:
    payload=json.loads(Path(output['file']).read_text())
    fps=payload['fps']['num']/payload['fps']['den']
    head_error=eye_error=0.
    for i,row in enumerate(payload['blendShapes']):
        index=int(np.argmin(abs(times-i/fps)));t=times[index]
        source=angles('performance_pose',['pitch_deg','roll_deg','yaw_deg'],t)
        actual_head=Euler(tuple(math.radians(a) for a in source),'XYZ').to_matrix()
        source=angles('performance_gaze',['yaw_deg','pitch_deg'],t)
        actual_eye=Matrix.Rotation(math.radians(source[0]),3,'Z') @ Matrix.Rotation(math.radians(source[1]),3,'X')
        head_error=max(head_error,float(np.max(abs(np.array(actual_head)-np.array(restore(*row[52:55]))))))
        for span in [row[55:58],row[58:61]]:
            eye_error=max(eye_error,float(np.max(abs(np.array(actual_eye)-np.array(restore(*span))))))
    assert head_error<1e-6 and eye_error<1e-6,(head_error,eye_error)
    checks.append({'fps':fps,'frames':len(payload['blendShapes']),
       'head_matrix_max_error':head_error,'eye_matrix_max_error':eye_error,
       'result':'passed','implementation':'Blender mathutils, matching render_variant.py; float32 tolerance 1e-6'})
report['independent_blender_rotation_check']=checks
report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(checks,indent=2))
