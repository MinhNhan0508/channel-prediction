"""Bounded retries when Windows temporarily prevents an atomic file replacement."""
import errno
import os
import time


def replace_with_retry(source, destination, attempts=101, interval=0.1):
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError as error:
            sharing_conflict = getattr(error, 'winerror', None) in {5, 32, 33}
            access_error = error.errno in {errno.EACCES, errno.EPERM}
            if not (sharing_conflict or access_error) or attempt + 1 == attempts:
                raise
            time.sleep(interval)
