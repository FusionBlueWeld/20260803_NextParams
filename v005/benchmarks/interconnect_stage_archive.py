"""Losslessly archive older response CSVs of completed new stage trials only."""
import gzip
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]/'v005/trials'


def sha(path, compressed=False):
    h=hashlib.sha256()
    with (gzip.open(path,'rb') if compressed else path.open('rb')) as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stages',nargs='+',default=['cut_cu','cut_al','form_cu','form_al','assembly','welding'])
    args=parser.parse_args()
    saved=0;count=0
    for folder in sorted(ROOT.glob('trial_interconnect_stage_*_20261010_focus_on')):
        if not (folder/'summary.json').is_file(): continue
        if json.loads((folder/'summary.json').read_text(encoding='utf-8'))['simulator'] not in args.stages: continue
        assert folder.resolve().is_relative_to(ROOT.resolve())
        space=folder/'output/response_spaces'
        files=sorted(space.glob('hybrid_response_space_run_*.csv'))
        manifest_path=folder/'response_archive_manifest.json'
        manifest=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
        for path in files[:-1]:
            assert path.resolve().parent==space.resolve()
            target=path.with_suffix('.csv.gz')
            assert not target.exists(),target
            original_hash=sha(path);original_size=path.stat().st_size
            with path.open('rb') as source, gzip.open(target,'wb',compresslevel=1) as dest:
                shutil.copyfileobj(source,dest,1024*1024)
            assert sha(target,compressed=True)==original_hash, 'Compression content mismatch'
            manifest[path.name]={'content_sha256':original_hash,'original_bytes':original_size,
                                 'archive':target.name,'archive_sha256':sha(target),'archive_bytes':target.stat().st_size}
            manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
            # The exact original bytes remain recoverable in the verified .gz in this same directory.
            path.unlink()
            saved+=original_size-target.stat().st_size;count+=1
        print(json.dumps({'trial':folder.name,'archived':len(manifest)}),flush=True)
    print(json.dumps({'new_archives':count,'bytes_saved':saved}),flush=True)


if __name__=='__main__': main()
