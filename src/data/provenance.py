"""Require an explicit observation policy before training portfolio models."""
from pathlib import Path
import hashlib
import json
from src.data.observation import OBSERVATION_POLICY


def data_content_fingerprint(data_dir: Path) -> str:
    """Fingerprint current NPZ bytes, including normalizer, in path order."""
    data_dir=Path(data_dir)
    digest=hashlib.sha256()
    for path in sorted(data_dir.rglob('*.npz')):
        digest.update(path.relative_to(data_dir).as_posix().encode('utf-8'))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def dataset_provenance(data_dir: Path) -> dict:
    path=Path(data_dir)/'metadata.json'
    if not path.is_file():
        raise ValueError('Dataset has no metadata.json; regenerate with the history-only generator.')
    raw=path.read_bytes()
    metadata=json.loads(raw)
    if metadata.get('schema_version') != 1:
        raise ValueError('Unsupported dataset metadata schema.')
    if metadata.get('status') != 'complete':
        raise ValueError('Dataset generation is incomplete.')
    if metadata.get('observation_policy') != OBSERVATION_POLICY:
        raise ValueError('Dataset observation policy does not match this experiment.')
    return {'manifest_sha256':hashlib.sha256(raw).hexdigest(),
            'observation_policy':OBSERVATION_POLICY}
