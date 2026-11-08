"""Emits SPIR-V for every environment of a precompiled file, one at a time, so a shader that makes
the compiler abort is found and listed instead of ending the run. Resumes after each crash.

usage: python -I emit_check.py <nx2-precompile.exe> <file.nx2prep> <crashes.json>
Writes {"crashes": [{"index", "stage", "code_hash"}...]}.
"""
import json
import os
import subprocess
import sys

tool, path, out_path = sys.argv[1:4]
progress = os.path.abspath(out_path + '.progress')
crashes = []
start = 0
while True:
    env = dict(os.environ, NX2_CHECK_EMIT='1', NX2_CHECK_PROGRESS=progress, NX2_CHECK_START=str(start))
    if os.path.exists(progress):
        os.remove(progress)
    run = subprocess.run([tool, 'check', '--file', path, '--threads', '1'], env=env,
                         capture_output=True, text=True)
    if 'environments ' in run.stdout:
        print(run.stdout.splitlines()[0])
        break
    index, stage, code_hash = open(progress).read().split()
    crashes.append(dict(index=int(index), stage=stage, code_hash=code_hash, exit=run.returncode))
    print('crash at environment', index, stage, code_hash, 'exit', run.returncode, flush=True)
    start = int(index) + 1
json.dump(dict(crashes=crashes), open(out_path, 'w'), indent=1)
print('crashing environments:', len(crashes))
