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

**All seven milestones complete.** All seven deliverable figures render.

### Headline result

Three seeds, both architectures, converged with early stopping on real BETSE data:

| split | MLP (26,625 params, no graph) | MPNN (663,553 params) | difference |
|---|---|---|---|
| `test_id` | 0.7974 ± 0.0161 mV | 0.7669 ± 0.0116 mV | −3.8% |
| `test_ood` | 1.1900 ± 0.0529 mV | 1.1615 ± 0.0567 mV | −2.4% |

Both clear the accuracy criterion (10% of range ≈ 8 mV) by roughly an order of magnitude. But on
the out-of-distribution split the paired per-seed difference **changes sign** (−0.052, +0.088,
−0.121), so the graph network is not distinguishable from the graph-blind baseline where it
matters most.

The cause is in the data, not the model: **every cell in every training tissue carries an
identical channel-density vector**, so ~99% of Vmem variance is between tissues rather than within
them, and the gap-junction term vanishes in the bulk for any conductance. The hypothesis that
topology is necessary is not refuted — it is untested, because this dataset cannot express it.

### Validation against real measurements

Curated 29 verified published Vmem measurements (0 fabricated) and tested the model differentially:
match a baseline ensemble to a measured control potential, apply the perturbation, compare predicted
against measured shift.

| stratum | n | MAE | vs 5.78 mV threshold |
|---|---|---|---|
| Channel blockade on a representable channel | 2 | 5.35 mV | meets |
| **Gap-junction blockade** | 3 | 9.80 mV | **fails** |
| All records | 6 | 12.46 mV | fails |

**All three gap-junction experiments predict ≈ 0.** Complete uncoupling, measured at +18.8 mV, is
predicted at +0.017 mV — the model learned that gap junctions do not matter, because in its
training data they did not. This failure was predicted in advance from the dataset diagnostic
above, then confirmed against measurements the model never saw.

Full analysis: [`logs/research-report.md`](logs/research-report.md).
Departures from the spec: [`DEVIATIONS.md`](DEVIATIONS.md).
Build narrative: [`logs/director-log.md`](logs/director-log.md).

### Dataset

12,000 BETSE simulations from 13,800 attempted, zero failures
(8000 train / 1000 val / 1000 test_id / 2000 test_ood, the last being 500 each of four
perturbation families). Details and caveats: [`data/synthetic/DATASET_CARD.md`](data/synthetic/DATASET_CARD.md).

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
