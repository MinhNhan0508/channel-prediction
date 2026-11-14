from pathlib import Path
import contextlib,io,json,tempfile,unittest
import torch
import numpy as np
from tools.train_suite import run_suite,single_writer

ROOT=Path(__file__).resolve().parents[1]


class SuiteTest(unittest.TestCase):
    def test_full_queue_evaluates_and_skips_completed_run_on_resume(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(prefix='mimo-suite-') as tmp:
            folder=Path(tmp)
            data=folder/'data';data.mkdir()
            counts=dict(train=8,val=4,**{f'test_v{v}':2 for v in [30,50,70,90,120]})
            rng=np.random.default_rng(7)
            for split,count in counts.items():
                (data/split).mkdir()
                for index in range(count):
                    history=rng.normal(size=(10,8,4,2)).astype(np.float32)
                    future=np.repeat(history[-1:],5,axis=0)
                    np.savez(data/split/f'sample_{index:05d}.npz',He_history=history,He_future=future,
                             velocity=30.,snr=20.,delay_spread=50.)
            np.savez(data/'normalize_stats.npz',scale=1.)
            (data/'metadata.json').write_text(json.dumps(dict(schema_version=1,status='complete',
                observation_policy='history_only_v1',split_counts=counts)),encoding='utf-8')
            recipe=folder/'train.yaml'
            recipe.write_text('n_epochs: 2\nbatch_size: 4\nlr: 0.001\nweight_decay: 0.0001\npatience: 5\nnum_workers: 0\nuse_amp: false\ncache_dataset: true\nloss_type: nmse_only\n',encoding='utf-8')
            plan=dict(id='unit_suite',data_dir=str(data),output_dir=str(folder/'output'),
                      device='cpu',require_cuda=False,train_config=str(recipe),seeds=[42],
                      variants=[dict(id='dast',model='pi_dast',model_config='configs/model/pi_dast.yaml')],
                      runs=[dict(variant='dast',seed=42)],test_velocities=[30,50,70,90,120],
                      expected_split_counts=counts)
            path=folder/'plan.json';path.write_text(json.dumps(plan),encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                self.assertTrue(run_suite(path))
                before=(folder/'output/runs/dast_seed42/checkpoints/checkpoint_best.pt').read_bytes()
                self.assertTrue(run_suite(path,resume=True))
            status=json.loads((folder/'output/status.json').read_text(encoding='utf-8'))
            self.assertEqual(status['state'],'complete')
            self.assertEqual(status['completed'],['dast_seed42'])
            self.assertEqual(before,(folder/'output/runs/dast_seed42/checkpoints/checkpoint_best.pt').read_bytes())
            result=json.loads((folder/'output/evaluation.json').read_text(encoding='utf-8'))
            self.assertEqual(len(result['individual']),5)

    def test_second_writer_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='mimo-lock-') as tmp:
            path=Path(tmp)/'lock'
            with single_writer(path),self.assertRaises((RuntimeError,BlockingIOError)):
                with single_writer(path):pass
