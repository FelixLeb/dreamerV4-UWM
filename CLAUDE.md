# CLAUDE.md

## Read this first

This repository is in a **highly exploratory research phase**. Ideas, training
protocols, architectures, and workflows change drastically and often.

**Do not treat any prose about project goals, "current focus", "best recipe",
training-mode framing, or collaboration workflow as authoritative — including
earlier versions of this file and the project memory.** Importing such framing
injects stale bias and produces misconceptions about what is actually being
built or trained right now.

- For **intent and direction**, rely on the **current conversation**.
- For **facts about the code**, **read the code now** and verify before
  asserting. If a memory or this file names a file/function/flag, confirm it
  still exists and still does what's claimed.
- If anything below conflicts with the repo or with the user, the repo and the
  conversation win — and this file should be corrected.

The notes below are intentionally limited to stable, verifiable facts.

## Stable repo facts (still: verify against current code)

- Package `dreamerv4uwm/`: models in `models/`, data in `datasets.py`, losses in
  `loss.py` / `loss_new.py`, samplers in `sampling*.py` and `inference/`.
  Training entrypoints in `scripts/`, Hydra configs in `scripts/config/`, SLURM
  scripts in `hpc/slurms/`.
- There is **more than one** training script and **more than one** loss module.
  Always check which one a given run/slurm actually uses before assuming.
- Training is Hydra + `torchrun` (DDP). Tokenizer is frozen during dynamics
  training.
- Data is sharded HDF5; loader `kind` selects `sharded_hdf5` vs `g1_chunked`
  (and others). Confirm the dataset's kind from its files (`shard_*.h5` vs
  `chunk_*.h5`) and the slurm overrides.
- No test suite or lint config in the repo — don't invent commands.
- To run Python in the project's training runtime (NYU HPC Singularity + conda),
  see the `run-python-in-container` memory for the exact invocation (it needs the
  ext3 `--overlay`; GPU nodes require an active SLURM job to ssh into).

## Working with the user

- Propose and discuss; don't restructure or "clean up" code unless asked.
- When a run finishes or a hypothesis is settled, offer to log a terse entry in
  `docs/experiments.md` (the in-repo experiment log).
