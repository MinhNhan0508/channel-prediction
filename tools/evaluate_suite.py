"""Evaluate every declared checkpoint; never silently skip a missing seed."""
from pathlib import Path
import argparse,csv,hashlib,json,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data.normalize import ChannelNormalizer
from src.data.provenance import dataset_provenance
from src.models.utils import build_model,is_spatial_model
from src.utils.config import load_config


def evaluate_suite(plan_path):
    plan=json.loads(Path(plan_path).read_text(encoding='utf-8'))
    data=(ROOT/plan['data_dir']).resolve();out=(ROOT/plan['output_dir']).resolve()
    experiment=json.loads((out/'experiment.json').read_text(encoding='utf-8'))
    provenance=dataset_provenance(data)
    if provenance!=experiment['data_provenance']:raise ValueError('Dataset identity changed.')
    from src.data.provenance import data_content_fingerprint
    metadata=json.loads((data/'metadata.json').read_text(encoding='utf-8'))
    if plan.get('require_data_integrity') and data_content_fingerprint(data)!=metadata.get('data_content_sha256'):
        raise ValueError('Dataset bytes changed since generation.')
    for name,digest in experiment['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Experiment source changed: {name}')
    variants={v['id']:v for v in plan['variants']}
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    norm=ChannelNormalizer.load(data/'normalize_stats.npz')
    tests={}
    for velocity in plan['test_velocities']:
        history=[];future=[]
        for path in sorted((data/f'test_v{velocity}').glob('sample_*.npz')):
            with np.load(path,allow_pickle=False) as sample:
                history.append(sample['He_history']);future.append(sample['He_future'])
        tests[velocity]=(np.stack(history),np.stack(future))
    records=[]
    for run in plan['runs']:
        variant=variants[run['variant']];run_id=f'{variant["id"]}_seed{run["seed"]}'
        run_dir=out/'runs'/run_id
        if not (run_dir/'training.json').is_file():raise FileNotFoundError(f'Missing finished run: {run_id}')
        training=json.loads((run_dir/'training.json').read_text(encoding='utf-8'))
        path=run_dir/'checkpoints/checkpoint_best.pt'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=training['checkpoint_sha256']:raise ValueError('Checkpoint hash changed.')
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved['config']['data_provenance']!=provenance:raise ValueError('Checkpoint dataset differs.')
        cfg=load_config(ROOT/variant['model_config'])
        model=build_model(variant['model'],cfg).to(device).eval()
        model.load_state_dict(saved['model_state_dict'],strict=True)
        raw={}
        for velocity,(history,true) in tests.items():
            preds=[]
            with torch.inference_mode():
                for first in range(0,len(history),64):
                    x=torch.from_numpy((history[first:first+64]/norm.scale).astype(np.float32)).to(device)
                    if not is_spatial_model(variant['model']):x=x.flatten(2)
                    preds.append(model(x).reshape(-1,5,8,4,2).cpu().numpy()*norm.scale)
            pred=np.concatenate(preds)
            if not np.isfinite(pred).all():raise FloatingPointError('Nonfinite test prediction.')
            values=((pred-true)**2).sum((2,3,4))/(true**2).sum((2,3,4)).clip(1e-12)
            raw[f'nmse_v{velocity}']=values
            records.append(dict(variant=variant['id'],seed=run['seed'],velocity_kmh=velocity,
                                n=len(true),nmse_db=float(10*np.log10(values.mean()))))
        np.savez(run_dir/'test_nmse.npz',**raw)
        del model,saved
    aggregate=[]
    for variant in variants:
        for velocity in plan['test_velocities']:
            group=[r for r in records if r['variant']==variant and r['velocity_kmh']==velocity]
            if len(group)!=len(plan['seeds']):raise ValueError('Incomplete seed group.')
            aggregate.append(dict(variant=variant,velocity_kmh=velocity,seeds=plan['seeds'],
                                  mean_db=float(np.mean([r['nmse_db'] for r in group])),
                                  std_population_db=float(np.std([r['nmse_db'] for r in group]))))
    payload=dict(scope='Synthetic CDL-A, history-only observation policy',experiment=experiment,
                 individual=records,aggregate=aggregate)
    (out/'evaluation.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    with (out/'evaluation.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print(f'[evaluate] {len(plan["runs"])} checkpoints, all declared seeds evaluated.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True)
    evaluate_suite((ROOT/p.parse_args().plan).resolve())
