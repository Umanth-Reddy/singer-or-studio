# Reproducing the method reruns on this machine

The JSON outputs are from runs made in `/tmp/credible-method-recheck`. Dataset and benchmark inputs stayed in `/legacy/ps`; they were not copied. To repeat a run, create the same scratch layout, copy `scripts/*.py` into their respective experiment directories, copy `pilots/` into scratch root, and make read-only input links:

```sh
ln -s /legacy/ps/glass-breaker/q2-study/data /tmp/credible-method-recheck/data
ln -s /legacy/ps/glass-breaker/q2-study/TEMP /tmp/credible-method-recheck/TEMP
```

Set `m1.DATA` in each script to `/legacy/ps/claude-research/data` (this is how the copied runs were executed). Run with `/legacy/ps/.venv-ssl/bin/python` and `OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2`. Outputs should go only to the scratch copy's `exploratory_*/results.json` files. Do not run unmodified older scripts from the original project path: several output paths are hardcoded.
