"""Run a fixed experiment matrix sequentially, with epoch-boundary resume."""
from pathlib import Path
import argparse
import contextlib
import hashlib
import json
import os
import platform
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data.dataset import ChannelDataset
from src.data.normalize import ChannelNormalizer
from src.data.provenance import dataset_provenance
from src.losses.combined import build_loss
from src.models.utils import build_model,is_spatial_model,count_trainable_params
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.logger import Logger
from src.utils.seed import set_seed
from src.utils.atomic_io import replace_with_retry


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    replace_with_retry(temporary,path)


@contextlib.contextmanager
def single_writer(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as stream:
        stream.seek(0,2)
        if stream.tell()==0:stream.write(b'0');stream.flush()
        stream.seek(0)
        if os.name=='nt':
            import msvcrt
            try:msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            except OSError:raise RuntimeError('Another suite process already owns this output directory.')
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            stream.seek(0)
            if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


class SuiteLogger(Logger):
    def __init__(self,log_dir,status,callback,run_id):
        super().__init__(log_dir)
        self.status=status;self.callback=callback;self.run_id=run_id
        self.previous=time.perf_counter()

    def log_metrics(self,metrics,step):
        if not all(np.isfinite(v) for v in metrics.values()):
            raise FloatingPointError('Nonfinite epoch metric.')
        super().log_metrics(metrics,step)
        now=time.perf_counter()
        self.status.update(state='training',active_run=self.run_id,epoch=step+1,
                           epoch_s=now-self.previous,val_nmse_db=float(10*np.log10(metrics['val/nmse_loss'])),
                           pid=os.getpid(),updated_unix_s=time.time())
        self.previous=now
        self.callback()


def run_suite(plan_path,resume=False):
    plan_path=Path(plan_path).resolve()
    plan=json.loads(plan_path.read_text(encoding='utf-8'))
    data=(ROOT/plan['data_dir']).resolve();out=(ROOT/plan['output_dir']).resolve()
    out.mkdir(parents=True,exist_ok=True)
    stop=out/'STOP'
    with single_writer(out/'suite.lock'):
        if stop.exists():
            if not resume:raise RuntimeError('STOP exists. Use run --resume to explicitly continue.')
            stop.unlink()
        requested=plan.get('device','auto')
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') if requested=='auto' else torch.device(requested)
        if plan.get('require_cuda') and device.type!='cuda':
            raise RuntimeError('This plan requires CUDA; refusing to silently start CPU training.')
        provenance=dataset_provenance(data)
        metadata=json.loads((data/'metadata.json').read_text(encoding='utf-8'))
        for split,count in plan['expected_split_counts'].items():
            if len(list((data/split).glob('sample_*.npz')))!=count:
                raise ValueError(f'Unexpected sample count: {split}')
        if plan.get('require_data_integrity'):
            from src.data.provenance import data_content_fingerprint
            if metadata.get('data_content_sha256')!=data_content_fingerprint(data):
                raise ValueError('Dataset files differ from the sealed generation manifest.')
        source={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                for folder in ['src','tools'] for p in sorted((ROOT/folder).rglob('*.py'))}
        variants={v['id']:v for v in plan['variants']}
        configs={k:load_config(ROOT/v['model_config']) for k,v in variants.items()}
        recipe=load_config(ROOT/plan['train_config'])
        manifest=dict(schema_version=1,plan=plan,training_config=recipe,models=configs,
                      data_provenance=provenance,source_sha256=source,
                      runtime=dict(torch=torch.__version__,numpy=np.__version__,python=platform.python_version(),
                                   device=str(device),gpu=torch.cuda.get_device_name() if device.type=='cuda' else None))
        manifest_bytes=json.dumps(manifest,sort_keys=True).encode('utf-8')
        manifest_hash=hashlib.sha256(manifest_bytes).hexdigest()
        manifest_path=out/'experiment.json'
        if manifest_path.exists():
            if json.loads(manifest_path.read_text(encoding='utf-8'))!=manifest:
                raise ValueError('Existing experiment identity differs; use a new output directory.')
        else:write_json(manifest_path,manifest)
        status=dict(state='starting',plan_id=plan['id'],total_runs=len(plan['runs']),completed=[],
                    pid=os.getpid(),experiment_sha256=manifest_hash)
        def update():write_json(out/'status.json',status)
        update()
        norm=ChannelNormalizer.load(data/'normalize_stats.npz')
        try:
            for index,item in enumerate(plan['runs']):
                variant=variants[item['variant']];seed=int(item['seed'])
                run_id=f'{variant["id"]}_seed{seed}'
                run_dir=out/'runs'/run_id
                run_dir.mkdir(parents=True,exist_ok=True)
                cfg=dict(configs[variant['id']],**recipe,seed=seed,data_provenance=provenance,
                         experiment_sha256=manifest_hash,stop_file=str(stop),quiet_progress=True)
                status.update(state='loading',active_run=run_id,run_index=index+1,epoch=0,updated_unix_s=time.time())
                update()
                complete=run_dir/'training.json'
                if complete.exists():
                    record=json.loads(complete.read_text(encoding='utf-8'))
                    if record['experiment_sha256']!=manifest_hash:raise ValueError('Run identity differs.')
                    status['completed'].append(run_id);update();continue
                if stop.exists():status.update(state='stopped');update();return False
                last=run_dir/'checkpoints/checkpoint_last.pt'
                if last.exists() and not resume:raise RuntimeError('Partial run exists. Use --resume.')
                set_seed(seed)
                train=ChannelDataset(data/'train',norm,cache_in_memory=bool(cfg.get('cache_dataset')))
                val=ChannelDataset(data/'val',norm,cache_in_memory=bool(cfg.get('cache_dataset')))
                train_loader=DataLoader(train,batch_size=cfg['batch_size'],shuffle=True,num_workers=0)
                val_loader=DataLoader(val,batch_size=cfg['batch_size'],num_workers=0)
                model=build_model(variant['model'],configs[variant['id']])
                trainer=Trainer(model,train_loader,val_loader,build_loss(cfg),device,cfg,
                                run_dir/'logs',run_dir/'checkpoints',flatten_input=not is_spatial_model(variant['model']))
                trainer.logger.close()
                trainer.logger=SuiteLogger(run_dir/'logs',status,update,run_id)
                if last.exists():trainer.load_checkpoint(last)
                if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
                started=time.perf_counter()
                print(f'[suite] {index+1}/{len(plan["runs"])} {run_id}',flush=True)
                trainer.train()
                if trainer.stopped:
                    status.update(state='stopped');update();return False
                if not trainer.training_complete:raise RuntimeError('Training returned without completion or stop.')
                saved=torch.load(run_dir/'checkpoints/checkpoint_best.pt',map_location='cpu',weights_only=True)
                result=dict(run_id=run_id,variant=variant['id'],model=variant['model'],seed=seed,
                            best_epoch=saved['epoch']+1,best_val_nmse_linear=trainer.best_val_loss,
                            final_epoch=torch.load(last,map_location='cpu',weights_only=True)['epoch']+1,
                            params=count_trainable_params(model),current_process_train_s=time.perf_counter()-started,
                            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
                            experiment_sha256=manifest_hash,
                            checkpoint_sha256=hashlib.sha256((run_dir/'checkpoints/checkpoint_best.pt').read_bytes()).hexdigest())
                write_json(complete,result)
                status['completed'].append(run_id);update()
                del trainer,model,saved,train,val,train_loader,val_loader
                if device.type=='cuda':torch.cuda.empty_cache()
            status.update(state='training_complete',active_run=None,updated_unix_s=time.time());update()
            if plan.get('evaluate_when_complete',True):
                from tools.evaluate_suite import evaluate_suite
                status.update(state='evaluating');update()
                evaluate_suite(plan_path)
            status.update(state='complete',updated_unix_s=time.time());update()
            return True
        except BaseException as error:
            status.update(state='failed',error=f'{type(error).__name__}: {error}',updated_unix_s=time.time())
            update();raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['run','status','stop'])
    parser.add_argument('--plan',required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    path=(ROOT/args.plan).resolve();plan=json.loads(path.read_text(encoding='utf-8'))
    out=(ROOT/plan['output_dir']).resolve()
    torch.set_num_threads(2)
    if args.command=='status':
        print((out/'status.json').read_text(encoding='utf-8') if (out/'status.json').exists() else '{"state":"not_started"}')
    elif args.command=='stop':
        out.mkdir(parents=True,exist_ok=True);(out/'STOP').write_text('Stop after the next saved epoch.\n',encoding='utf-8')
        print('Stop requested. Inspect status before closing the process.')
    else:run_suite(path,resume=args.resume)


if __name__=='__main__':main()
