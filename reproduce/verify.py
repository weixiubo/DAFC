"""Rebuild all C588 estimates and optionally verify published attempt records and images."""
import argparse, hashlib, json
from collections import defaultdict
from pathlib import Path
from frozen_statistics import *
from dafc.evaluation.attempts import sequence_errors

HERE=Path(__file__).resolve().parent
def load(p): return json.loads(p.read_text(encoding="utf-8-sig"))
def verify(raw_root=None):
    for name, expected in load(HERE/'checksums.json').items():
        assert sha256(HERE/name)==expected, f"publication content changed: {name}"
    rows=load(HERE/'rows.json');expected=load(HERE/'expected.json')
    assert len(rows)==1120 and len({r['job_id'] for r in rows})==1120
    assert all(not r['authorized'] for r in rows if not r['source_success'])
    assert all(r['model_calls']==1 for r in rows if r['policy']=='always')
    actual={a:{p:policy_summary(rows,a,p) for p in POLICIES} for a in ACTORS}
    assert actual==expected['summary'], 'policy summaries changed'
    comparisons={};seed=BOOTSTRAP_SEED
    for actor in ACTORS:
        comparisons[actor]={}
        for right in ('always','s_only','standard_ransac'):
            comparisons[actor]['dafc_minus_'+right]=comparison(rows,actor,'dafc',right,seed)
            seed+=1
    assert comparisons==expected['comparisons'], 'paired estimates or bootstrap intervals changed'
    if raw_root is not None:
        root=Path(raw_root).resolve()
        if not root.is_dir():raise FileNotFoundError(f"Supply the published C587-v3 attempt directory: {root}")
        files=load(HERE/'raw_manifest.json')['files']
        for record in files:
            path=(root/record['path']).resolve()
            assert root in path.parents and sha256(path)==record['sha256'], record['path']
        lifecycle=defaultdict(list);outcomes={}
        for path in sorted((root/'results').rglob('*.json')):
            payload=load(path);lifecycle[payload['job']['job_id']].append(payload)
            if payload.get('outcome_bearing'):
                item=compact_row(payload,path);item['attempt_path']=str(path.relative_to(root))
                assert item['job_id'] not in outcomes
                outcomes[item['job_id']]=item
            for key in ('before','before_raw','after'):
                image=payload.get(key)
                if not image:continue
                parts=Path(image['path']).parts
                pos=parts.index('images')
                image_path=root/Path(*parts[pos:])
                assert sha256(image_path)==image['sha256'], f"image receipt mismatch: {image_path}"
        assert len(lifecycle)==1120
        for attempts in lifecycle.values():
            assert not sequence_errors(sorted(attempts,key=lambda x:x['attempt']),require_complete=True)
        assert outcomes=={r['job_id']:r for r in rows}, 'raw-to-public projection changed'
        print(f"Verified {len(files)} published files, 1121 attempts, 3360 image receipts and public projections")
    print('Verified 1120 outcomes; rebuilt 8 policy summaries and 6 paired 20000-sample bootstrap comparisons')
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root',type=Path)
    verify(parser.parse_args().raw_root)
