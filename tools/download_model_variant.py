"""Download a pinned NVIDIA model snapshot with resumable ranges and SHA256 checks."""
import argparse,concurrent.futures,hashlib,json,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent.parent
parser=argparse.ArgumentParser();parser.add_argument('repo');parser.add_argument('folder')
parser.add_argument('--refresh',action='store_true',help='Explicitly update the pinned model snapshot')
args=parser.parse_args()
folder=root/'assets/models'/args.folder;folder.mkdir(parents=True,exist_ok=True)
manifestdir=root/'tools/model-manifests';manifestdir.mkdir(exist_ok=True)
def get_json(url):return json.loads(subprocess.check_output(['curl','--http1.1','-fsSL','--retry','4',url]))
manifest_path=manifestdir/(args.folder+'.json')
if manifest_path.exists() and not args.refresh:
    snapshot=json.loads(manifest_path.read_text())
    if snapshot['repo']!=args.repo:raise ValueError('Repository differs from pinned model manifest')
    revision=snapshot['revision'];manifest=snapshot['files']
else:
    info=get_json('https://huggingface.co/api/models/'+args.repo)
    revision=info['sha'];manifest=get_json('https://huggingface.co/api/models/'+args.repo+'/tree/'+revision)
    manifest_path.write_text(json.dumps({'repo':args.repo,'revision':revision,'files':manifest},indent=2)+'\n')
for item in manifest:
    name=item['path']
    if item['type']!='file' or name=='.gitattributes':continue
    dest=folder/name;size=item['size']
    if not dest.exists() or dest.stat().st_size!=size:
        print('Download',name,size,flush=True)
        url='https://huggingface.co/'+args.repo+'/resolve/'+revision+'/'+name
        workers=8 if size>16<<20 else 1
        parts=[(Path(str(dest)+f'.part{i}'),size*i//workers,size*(i+1)//workers-1) for i in range(workers)]
        def fetch(part):
            path,a,b=part
            if size==0:path.write_bytes(b'');return
            for attempt in range(8):
                have=path.stat().st_size if path.exists() else 0
                if have==b-a+1:return
                assert have<b-a+1
                tail=Path(str(path)+'.tail')
                result=subprocess.run(['curl','--http1.1','-fLsS','--connect-timeout','30','--max-time','180','--range',f'{a+have}-{b}',url,'-o',str(tail)])
                if tail.exists() and tail.stat().st_size:
                    assert tail.stat().st_size<=b-a+1-have,(name,'server ignored requested range')
                    with path.open('ab') as output,tail.open('rb') as source:
                        while chunk:=source.read(8<<20):output.write(chunk)
                    tail.unlink()
                if result.returncode:time.sleep(2)
            assert path.stat().st_size==b-a+1,(name,'incomplete download')
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(fetch,parts))
        with dest.open('wb') as output:
            for part,_,_ in parts:
                with part.open('rb') as source:
                    while chunk:=source.read(8<<20):output.write(chunk)
                part.unlink()
    assert dest.stat().st_size==size,name
    if item.get('lfs',{}).get('oid'):
        with dest.open('rb') as source:
            sha=hashlib.file_digest(source,'sha256') if hasattr(hashlib,'file_digest') else None
            if sha is None:
                sha=hashlib.sha256()
                while chunk:=source.read(8<<20):sha.update(chunk)
        assert sha.hexdigest()==item['lfs']['oid'],name
    print('Verified',name,flush=True)
print('MODEL_READY',args.folder,revision,flush=True)
