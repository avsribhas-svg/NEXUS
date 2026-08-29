# NEXUS — Phase 1

Predicting bioelectric state from molecular specification.

Phase 1 asks the smallest honest version of the question: can a learned model predict per-cell
membrane voltage across a tissue, given only each cell's ion-channel densities and the gap-junction
wiring between cells? Ground truth comes from [BETSE](https://github.com/betsee/betse), a
bioelectric physics simulator. The model is a message-passing graph neural network — cells are
nodes, gap junctions are edges — chosen because message passing is the learned analogue of current
spreading through gap junctions.

Full specification: [`nexus-phase1-prd.md`](nexus-phase1-prd.md).

## Status

**92/92 tests passing**, including all five BETSE integration tests.

```
tests/test_data_pipeline.py ..............................  [ 32%]
tests/test_evaluation.py    .............                   [ 46%]
tests/test_integration.py   ........                        [ 55%]
tests/test_model.py         ........................        [ 81%]
tests/test_numerical.py     .......                         [ 89%]
tests/test_training.py      ..........                      [100%]
======================= 92 passed in 461.20s =======================
```

Dataset generation is in progress: 13,800 BETSE simulations for a 12,000-sample dataset
(8000 train / 1000 val / 1000 test_id / 2000 test_ood).

**The model has not yet been trained on real BETSE data.** Every test to date runs against
synthetic fixtures with a deterministic analytic target. Passing `test_gnn_beats_mlp_on_coupled_data`
shows the architecture can exploit graph structure; it does not show it can learn real bioelectrics.
That experiment begins when generation completes.

## Layout

```
nexus/data/        config sampling, BETSE driver, PyG dataset, validation
nexus/model/       MPNN, per-cell MLP baseline, losses
nexus/training/    trainer, config
nexus/evaluation/  metrics, generalization grouping, figures
scripts/           dataset generation and split finalization
tests/             the acceptance specification
logs/              director log — full build record and every design decision
data/synthetic/    DATASET_CARD.md — what the target variable actually is
```

## Running

```bash
pip install torch torch_geometric numpy scipy pandas scikit-learn matplotlib seaborn joblib pytest
pytest                        # 87 tests, skips BETSE-dependent ones
pip install betse             # requires Python >= 3.11
pytest -m "betse or not betse"   # all 92
```

Generation and finalization:

```bash
python scripts/generate_dataset.py --n-baseline 11500 --n-per-perturbation 575 --jobs 12
python scripts/finalize_dataset.py --staged data/synthetic/_staged --out data/synthetic
```

`generate_dataset.py` is resumable — every outcome is flushed to `results.csv` as it completes, and
re-invoking skips finished work.

## Read these before using the data

**[`data/synthetic/DATASET_CARD.md`](data/synthetic/DATASET_CARD.md)** — the target is stored under
the key `vmem_steady_state`, and **that name is inaccurate**. The quantity is *Vmem after 5 s
equilibration*. Measurement showed the system has no reachable steady state under random parameter
sampling: a fast electrical transient completes by ~4–6 s, then a slow secular concentration drift
from continuous pump activity takes over and never settles. The card documents this and five other
limitations, including two input channels that carry no signal by construction.

**[`logs/director-log.md`](logs/director-log.md)** — the full build record.

## How this was built

Every line of Python in `nexus/` and `scripts/` was written by **qwen2.5-coder:7b** running locally,
directed by Claude Opus. The director decomposes, specifies, runs the tests, and hands back raw
output; it writes no code. The protocol is in [`CLAUDE.md`](CLAUDE.md) and
[`claude-md-addendum.md`](claude-md-addendum.md).

The failure distribution turned out to be the interesting part. Most corrections traced to the
director's specifications rather than the model: a hardcoded forward-slash path that BETSE rejects
on Windows, a physics mapping onto a parameter that measurement showed was inert, an `int()` cast on
a string identifier. The 7B's own errors clustered in boilerplate — a dropped `except` clause, an
inverted cleanup block, six lines deleted when asked to insert beside them — never in the
message-passing math.

One deliberate experiment: after six consecutive first-attempt passes the scaffold was degraded to
a file path, one sentence, and the tests. The model produced five independent defects in
`trainer.py`. Re-issued at full scaffold, it was correct in one attempt. The convergence was a
property of the environment, not the weights.
