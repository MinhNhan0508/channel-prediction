"""Atomic checkpoints with experiment identity and epoch-boundary RNG state."""
from pathlib import Path
import os
import random
import numpy as np
import torch
from src.utils.atomic_io import replace_with_retry


def random_state() -> dict:
    name,keys,pos,has_gauss,cached=np.random.get_state()
    return {'python':random.getstate(),'numpy':{'name':name,'keys':keys.tolist(),
            'pos':pos,'has_gauss':has_gauss,'cached':cached},
            'torch_cpu':torch.get_rng_state(),
            'torch_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_random_state(state: dict) -> None:
    random.setstate(state['python'])
    n=state['numpy']
    np.random.set_state((n['name'],np.asarray(n['keys'],dtype=np.uint32),n['pos'],n['has_gauss'],n['cached']))
    torch.set_rng_state(state['torch_cpu'].cpu())
    if state['torch_cuda']:
        if len(state['torch_cuda'])!=torch.cuda.device_count():
            raise ValueError('Resume requires the same number of CUDA devices.')
        torch.cuda.set_rng_state_all([s.cpu() for s in state['torch_cuda']])


def atomic_save(payload: dict,path: Path) -> None:
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        torch.save(payload,stream)
        stream.flush()
        os.fsync(stream.fileno())
    replace_with_retry(temporary,path)


def load_for_resume(path: Path,expected_config: dict) -> dict:
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint.get('schema_version')!=2:
        raise ValueError('Checkpoint lacks resumable RNG/schema metadata; start a new experiment.')
    if checkpoint['config']!=expected_config:
        raise ValueError('Resume config differs: model, data provenance, seed and training recipe must match.')
    return checkpoint
