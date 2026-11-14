import errno
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.utils.atomic_io import replace_with_retry


class AtomicIOTest(unittest.TestCase):
    def test_transient_conflict_retries_without_removing_old_target(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / 'new', Path(folder) / 'old'
            source.write_text('new')
            target.write_text('old')
            original = os.replace
            calls = []
            def replace(src, dst):
                calls.append(1)
                if len(calls) == 1:
                    self.assertEqual(target.read_text(), 'old')
                    raise PermissionError(errno.EACCES, 'sharing conflict')
                return original(src, dst)
            with patch('src.utils.atomic_io.os.replace', replace), patch('src.utils.atomic_io.time.sleep'):
                replace_with_retry(source, target)
            self.assertEqual(target.read_text(), 'new')
            self.assertEqual(len(calls), 2)

    def test_permanent_error_has_a_retry_limit(self):
        with patch('src.utils.atomic_io.os.replace', side_effect=PermissionError(errno.EACCES, 'denied')), \
                patch('src.utils.atomic_io.time.sleep') as sleep:
            with self.assertRaises(PermissionError):
                replace_with_retry('new', 'old', attempts=3)
            self.assertEqual(sleep.call_count, 2)

    def test_disk_full_is_not_retried_or_hidden(self):
        with patch('src.utils.atomic_io.os.replace', side_effect=OSError(errno.ENOSPC, 'full')), \
                patch('src.utils.atomic_io.time.sleep') as sleep:
            with self.assertRaises(OSError):
                replace_with_retry('new', 'old')
            sleep.assert_not_called()
