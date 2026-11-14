from pathlib import Path
import contextlib
import io
import tempfile
import unittest
import torch
from torch.utils.data import DataLoader,Dataset
from src.training.trainer import Trainer
from src.training.checkpoint import atomic_save,load_for_resume
from src.losses.combined import NMSEOnlyLoss
from src.utils.seed import set_seed


class TinyData(Dataset):
    def __len__(self):return 7
    def __getitem__(self,index):
        x=torch.tensor([[[index/7+0.2,index/11+0.3]]]).reshape(1,2)
        return {'history':x,'future':0.7*x+0.1}


def trainer(folder,seed=33):
    set_seed(seed)
    model=torch.nn.Sequential(torch.nn.Linear(2,4),torch.nn.ReLU(),torch.nn.Dropout(.2),torch.nn.Linear(4,2))
    cfg=dict(n_epochs=3,batch_size=3,lr=.01,weight_decay=.0001,warmup_ratio=.05,
             patience=10,num_workers=0,use_amp=False,seed=seed,
             data_provenance={'observation_policy':'history_only_v1','manifest_sha256':'fixture'})
    return Trainer(model,DataLoader(TinyData(),batch_size=3,shuffle=True),
                   DataLoader(TinyData(),batch_size=3),NMSEOnlyLoss(),torch.device('cpu'),cfg,
                   folder/'logs',folder/'checkpoints')


class Interrupted(Exception):pass


class CheckpointTest(unittest.TestCase):
    def test_interrupted_resume_matches_uninterrupted_cpu_training(self):
        with tempfile.TemporaryDirectory(prefix='mimo-resume-') as tmp:
            folder=Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                whole=trainer(folder/'whole');whole.train()
                first=trainer(folder/'resumed')
                save=first._save_checkpoint
                def stop_after_first(epoch,tag):
                    save(epoch,tag)
                    if epoch==0 and tag=='last':raise Interrupted()
                first._save_checkpoint=stop_after_first
                with self.assertRaises(Interrupted):first.train()
                resumed=trainer(folder/'resumed')
                resumed.load_checkpoint(folder/'resumed/checkpoints/checkpoint_last.pt')
                resumed.train()
            for name,value in whole.model.state_dict().items():
                self.assertTrue(torch.equal(value,resumed.model.state_dict()[name]),name)
            self.assertEqual(whole.best_val_loss,resumed.best_val_loss)
            self.assertTrue(resumed.training_complete)

    def test_config_mismatch_is_rejected_before_loading_weights(self):
        with tempfile.TemporaryDirectory(prefix='mimo-config-') as tmp:
            path=Path(tmp)/'checkpoint.pt'
            atomic_save({'schema_version':2,'config':{'seed':42}},path)
            with self.assertRaisesRegex(ValueError,'config differs'):
                load_for_resume(path,{'seed':123})

    def test_legacy_checkpoint_is_not_silently_resumed(self):
        with tempfile.TemporaryDirectory(prefix='mimo-legacy-') as tmp:
            path=Path(tmp)/'checkpoint.pt'
            atomic_save({'config':{}},path)
            with self.assertRaisesRegex(ValueError,'schema'):
                load_for_resume(path,{})

    def test_stop_saves_a_resumable_epoch_without_marking_complete(self):
        with tempfile.TemporaryDirectory(prefix='mimo-stop-') as tmp:
            folder=Path(tmp);stop=folder/'STOP';stop.write_text('stop',encoding='utf-8')
            t=trainer(folder);t.config['stop_file']=str(stop)
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):t.train()
            saved=torch.load(folder/'checkpoints/checkpoint_last.pt',weights_only=True)
            self.assertEqual(saved['epoch'],0)
            self.assertFalse(saved['training_complete'])
            self.assertTrue(t.stopped)
