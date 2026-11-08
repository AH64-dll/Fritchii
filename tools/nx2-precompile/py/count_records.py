"""Prints a disk cache's record count (the next session's first new record).

usage: python -I count_records.py <cache.bin>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lichii_cache as lc  # noqa: E402

_, records = lc.parse_cache(open(sys.argv[1], 'rb').read(), mesh_fields=True)
print('records now', sum(1 for _ in records))
