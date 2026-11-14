from pathlib import Path
import sys
import json
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.provenance import dataset_provenance


class ProvenanceTest(unittest.TestCase):
    def check_metadata(self,metadata):
        with tempfile.TemporaryDirectory(prefix='mimo-provenance-') as folder:
            path=Path(folder)/'metadata.json'
            if metadata is not None:path.write_text(json.dumps(metadata),encoding='utf-8')
            return dataset_provenance(Path(folder))

    def test_complete_causal_manifest_has_fingerprint(self):
        result=self.check_metadata(dict(schema_version=1,status='complete',observation_policy='history_only_v1'))
        self.assertEqual(result['observation_policy'],'history_only_v1')
        self.assertEqual(len(result['manifest_sha256']),64)

    def test_legacy_dataset_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'no metadata'):self.check_metadata(None)

    def test_incomplete_generation_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'incomplete'):
            self.check_metadata(dict(schema_version=1,status='generating',observation_policy='history_only_v1'))

    def test_other_policy_or_schema_is_rejected(self):
        for policy,schema in [('oracle',1),('history_only_v1',2)]:
            with self.subTest(policy=policy,schema=schema),self.assertRaises(ValueError):
                self.check_metadata(dict(schema_version=schema,status='complete',observation_policy=policy))
