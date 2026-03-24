# Cross-Tokenizer On-Policy Distillation — Experiment Protocol

## Loop

```
LOOP (Step 1..18):
  1. Review: git log + results.tsv
  2. Pick: ROADMAP next uncompleted step
  3. Change: ONE focused change
  4. Commit: git commit -m "xdistill: step N - <desc>"
  5. Verify: local unit test or remote SLURM submission
  6. Decide: pass → keep; fail → fix (max 2 attempts) or revert
  7. Log: append to results.tsv
  8. Repeat
```

## Revert

```bash
git reset --hard HEAD~1
echo "<commit>\t<step>\trevert\t0\t<reason>" >> results.tsv
```
