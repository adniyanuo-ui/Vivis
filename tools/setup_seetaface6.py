#!/usr/bin/env python3
"""Build pinned SeetaFace6 CPU/NEON libraries in outputs, without system installs."""
import argparse,hashlib,json,platform,subprocess,urllib.request,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def call(*args,log):subprocess.run(list(map(str,args)),check=True,stdout=log,stderr=subprocess.STDOUT)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--jobs',type=int,default=4);args=p.parse_args()
    if platform.machine()!='aarch64' or not 1<=args.jobs<=12:p.error('This recipe is validated for native aarch64; jobs must be 1..12')
    m=json.loads((ROOT/'native/vision/seeta/sources.json').read_text());base=ROOT/'outputs/face_ab'
    src=base/'src/SeetaFace6Open';build=base/'build';prefix=src/'build';base.mkdir(parents=True,exist_ok=True)
    with (base/'setup.log').open('w') as log:
        fresh=not src.exists()
        if fresh:call('git','clone','--no-checkout',m['repository'],src,log=log)
        current=subprocess.check_output(['git','-C',str(src),'rev-parse','HEAD'],text=True).strip()
        if fresh or current!=m['commit']:
            if not fresh and subprocess.check_output(['git','-C',str(src),'status','--porcelain'],text=True).strip():raise ValueError('Source checkout has changes; will not overwrite')
            call('git','-C',src,'checkout','--detach',m['commit'],log=log)
        call('git','-C',src,'submodule','update','--init','--recursive',log=log)
        for name in ('gcc13.patch','brightness_roi.patch'):
            patch=ROOT/'native/vision/seeta'/name
            # Idempotent patches; fail on source drift rather than silent edits.
            already=subprocess.run(['git','-C',str(src),'apply','--reverse','--check',str(patch)],stdout=log,stderr=log).returncode==0
            if not already:call('git','-C',src,'apply',patch,log=log)
        for part in ('OpenRoleZoo','SeetaAuthorize','TenniS','FaceBoxes','Landmarker','FaceRecognizer6','PoseEstimator6','QualityAssessor3','SeetaAgePredictor'):
            print('Building '+part,flush=True)
            flags=['-DCMAKE_BUILD_TYPE=Release','-DCONFIGURATION=Release','-DPLATFORM=auto',
                   '-DCMAKE_INSTALL_PREFIX='+str(prefix),'-DORZ_ROOT_DIR='+str(prefix)]
            if part=='TenniS':flags+=['-DTS_ON_ARM=ON','-DTS_USE_NEON=ON','-DTS_USE_OPENMP=ON','-DTS_USE_CUDA=OFF','-DTS_USE_CUBLAS=OFF','-DTS_BUILD_TEST=OFF','-DTS_BUILD_TOOLS=OFF']
            call('cmake','-S',src/part,'-B',build/part,*flags,log=log)
            call('cmake','--build',build/part,'--parallel',args.jobs,log=log)
            call('cmake','--install',build/part,log=log)
        call('cmake','-S',ROOT/'native/vision/seeta','-B',build/'probe','-DSEETA_PREFIX='+str(prefix),'-DCMAKE_BUILD_TYPE=Release',log=log)
        call('cmake','--build',build/'probe','--parallel',args.jobs,log=log)
        call(build/'probe/seeta_quality_test',log=log)
    models=base/'models';models.mkdir(exist_ok=True);archive=models/'sf6.0_models.zip'
    valid=lambda path,row:path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    if not all(valid(models/row['name'],row) for row in m['models']):
        if not valid(archive,m['archive']):
            print('Downloading official model archive',flush=True);temporary=archive.with_suffix('.part')
            with urllib.request.urlopen(m['archive']['url'],timeout=60) as response,temporary.open('wb') as stream:
                while block:=response.read(1024*1024):
                    if stream.tell()+len(block)>m['archive']['bytes']:raise ValueError('Oversized model archive')
                    stream.write(block)
            if not valid(temporary,m['archive']):raise ValueError('Model archive hash mismatch')
            temporary.replace(archive)
        with zipfile.ZipFile(archive) as zipped:
            for row in m['models']:
                target=models/row['name'];target.write_bytes(zipped.read('sf3.0_models/'+row['name']))
                if not valid(target,row):raise ValueError('Model hash mismatch')
    patches={p.name:sha(p) for p in (ROOT/'native/vision/seeta').glob('*.patch')}
    (base/'build_manifest.json').write_text(json.dumps(dict(sources=m,patches=patches,architecture=platform.machine(),cpu_neon=True,cuda=False),indent=2))
    print('SeetaFace6 build/model verification complete; no devices opened or runtime switched.')
if __name__=='__main__':main()
