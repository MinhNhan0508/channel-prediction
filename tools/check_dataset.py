"""Check every generated sample and the sealed dataset-content fingerprint."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data.provenance import dataset_provenance,data_content_fingerprint
from src.data.normalize import ChannelNormalizer


def check_dataset(data):
    data=Path(data)
    provenance=dataset_provenance(data)
    meta=json.loads((data/'metadata.json').read_text(encoding='utf-8'))
    if data_content_fingerprint(data)!=meta.get('data_content_sha256'):
        raise ValueError('Dataset bytes do not match the generation manifest.')
    cfg=meta['config'];p,q=cfg['p_frames'],cfg['q_frames']
    r,m,b=cfg['n_rf'],cfg['n_m'],cfg['n_bs']
    expected={'train':cfg['n_train'],'val':cfg['n_val'],
              **{f'test_v{int(v)}':cfg['n_test_per_scenario'] for v in cfg['test_velocities']}}
    seen=set();counts={};train_max=0.
    for split,count in expected.items():
        files=sorted((data/split).glob('sample_*.npz'))
        if len(files)!=count:raise ValueError(f'Split count mismatch: {split}')
        for path in files:
            with np.load(path,allow_pickle=False) as sample:
                shapes={'He_history':(p,r,m,2),'He_history_clean':(p,r,m,2),
                        'He_future':(q,r,m,2),'D':(r,b,2)}
                for key,shape in shapes.items():
                    value=sample[key]
                    if value.shape!=shape or value.dtype!=np.float32 or not np.isfinite(value).all():
                        raise ValueError(f'Invalid {key}: {path}')
                fingerprint=hashlib.sha256(sample['He_history_clean'].tobytes()).digest()
                if fingerprint in seen:raise ValueError(f'Exact duplicate clean history: {path}')
                seen.add(fingerprint)
                if split=='train':train_max=max(train_max,float(np.abs(sample['He_history']).max()),float(np.abs(sample['He_future']).max()))
                if split.startswith('test_v'):
                    velocity=int(split[6:])
                    if not np.isclose(float(sample['velocity']),velocity):raise ValueError('Test velocity mismatch.')
                    if not np.isclose(float(sample['snr']),cfg['test_snr']):raise ValueError('Test SNR mismatch.')
        counts[split]=len(files)
    scale=ChannelNormalizer.load(data/'normalize_stats.npz').scale
    if not np.isclose(scale,train_max,rtol=1e-7):raise ValueError('Normalizer does not match train-only max-abs.')
    return dict(checked_samples=sum(counts.values()),split_counts=counts,all_shapes_and_values_valid=True,
                unique_clean_histories=len(seen),normalizer_train_only=True,
                normalizer_scale=scale,data_provenance=provenance,
                data_content_sha256=meta['data_content_sha256'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('data_dir');parser.add_argument('--output')
    args=parser.parse_args();result=check_dataset((ROOT/args.data_dir).resolve())
    if args.output:Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
