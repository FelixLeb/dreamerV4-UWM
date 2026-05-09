# Running a Jupyter server on an HPC node (singularity + conda)

When VSCode is already attached to the same HPC node, no SSH tunnel is needed — bind the server to `127.0.0.1` and point VSCode at the printed URL.

## 1. Enter the container

The usual launch sequence (master env vars, NCCL config, cache setup) ends with:

```bash
module purge

export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# --------------------------
# Scratch-contained caches (single folder, no auto-cleanup)
# --------------------------
export CACHE_ROOT="/scratch/$USER/.jobcache"
export SCRATCH_HOME="${CACHE_ROOT}/home"
export TMPDIR="${CACHE_ROOT}/tmp"
export APPTAINER_CACHEDIR="${CACHE_ROOT}/apptainer"
export SINGULARITY_CACHEDIR="${CACHE_ROOT}/singularity"

mkdir -p "${SCRATCH_HOME}" "${TMPDIR}" "${APPTAINER_CACHEDIR}" "${SINGULARITY_CACHEDIR}"

# --------------------------
# Container execution
# --------------------------
OVERLAY="/scratch/$USER/projects/dreamer-v4/hpc/overlay-25GB-500K.ext3:ro"
IMAGE="/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif"

export PYTHONPATH="/scratch/rk4342/projects/dreamerV4-UWM:${PYTHONPATH}"

singularity exec --nv --no-home \
  --bind "${SCRATCH_HOME}:$HOME" \
  --bind "/scratch/$USER:/scratch/$USER" \
  --overlay "${OVERLAY}" \
  --env TMPDIR="${TMPDIR}" \
  --env APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR}" \
  --env SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR}" \
  "${IMAGE}" \
  bash

source /ext3/env.sh
conda activate dreamerv4
```

If Jupyter is missing from the env, install it once. The overlay is mounted `:ro`, so either:

- temporarily drop `:ro` on `--overlay`, install, then re-mount `:ro`, **or**
- install into the bound `$HOME` with user site-packages:
  ```bash
  pip install --user jupyter ipykernel
  ```

(Optional) register the env as a named kernel for clearer kernel picking:

```bash
python -m ipykernel install --user --name dreamerv4 --display-name "Python (dreamerv4)"
```

## 2. Launch the server

```bash
jupyter server --no-browser --ip=127.0.0.1 --port=8888
```

If 8888 is already in use on this shared node, pick a random free port:

```bash
jupyter server --no-browser --ip=127.0.0.1 --port=$((20000 + RANDOM % 10000))
```

Copy the full `http://127.0.0.1:<port>/?token=...` URL it prints — token included.

To keep the server alive after closing the shell:

```bash
nohup jupyter server --no-browser --ip=127.0.0.1 --port=8888 \
  > $HOME/jupyter.log 2>&1 &
# token URL is in $HOME/jupyter.log
```

## 3. Connect VSCode

In the VSCode window attached to this node:

1. Open or create a `.ipynb`.
2. Click the kernel picker (top-right of the notebook).
3. **Select Another Kernel… → Existing Jupyter Server…**
4. Paste the `http://127.0.0.1:<port>/?token=…` URL.
5. Pick the **Python (dreamerv4)** kernel.

## Gotchas

- Server dies on shell exit unless backgrounded with `nohup` / `tmux`.
- Always bind to `127.0.0.1`, never `0.0.0.0` — other users on the shared node would otherwise have access.
- `pip install` against the read-only overlay will silently fail in confusing ways; use `--user` or remount rw.
