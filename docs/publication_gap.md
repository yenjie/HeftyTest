# Publication Gap Status

Historical note.

This file used to track the gap between the reduced-static benchmark and the
publication-level Tang/TAMU workflow with hard-coded metrics. Those metrics drift
whenever the saved benchmark branch is updated, so this file is no longer the
canonical source of truth.

Use [docs/code_structure.md](code_structure.md) for the current branch mapping
and workflow definitions.

For the live metrics of a saved branch, read:

- `results/task1/report.md`
- `results/task1_publication_faithful/report.md`
- `results/task1_publication_locked_potential/report.md`
- `results/task1_publication_smoothed_fig4/report.md`
- `results/task1_tang_exact/report.md`

The practical interpretation remains:

- the main saved `results/task1` branch is a publication-faithful reduced-static
  hybrid fit, not a literal Tang replay
- the Tang disagreement is therefore partly structural
- the cleanest diagnostic of that structural gap is the separate
  `task1-tang-exact` branch
