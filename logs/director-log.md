# NEXUS Phase 1: Director Log

Director: Claude Opus 5 (Claude Code, Mac). Executor: qwen2.5-coder:7b via Ollama on `abhiram-lenovo` (Windows, RTX 4050 Laptop 6GB).
Protocol: CLAUDE.md + claude-md-addendum.md (six architectural properties).

---

## [Session 1] Milestone 0: Infrastructure

### Environment findings (deviations from CLAUDE.md as written)

CLAUDE.md's connection details were placeholders. Actual, verified:

| CLAUDE.md assumed | Actual |
|---|---|
| `ssh windows-rig@abhiram-lenovo` | `ssh windows-rig` (SSH config alias -> `abhis@100.66.69.108`) |
| POSIX shell on Windows | Windows PowerShell 5.1; no `&&`, no heredocs, no `~/`, no brace expansion |
| Ollama needs starting | Already serving; reachable directly from the Mac at `http://abhiram-lenovo:11434`; no SSH hop needed for 7B calls |
| `pip --break-system-packages` | Not needed (native Windows Python 3.12.10, not Debian-managed) |

Consequences for the protocol:
- The CLAUDE.md heredoc pattern (`cat > file << 'PYEOF'`) cannot be used. Files are written locally on the Mac from the 7B's raw response, then `scp`'d to the rig. This is strictly more robust, no shell expansion of Python source at all.
- 7B calls go Mac -> Ollama HTTP directly. Lower latency, no SSH quoting layer.
- Remote commands are authored in PowerShell syntax.

Verified: Tailscale up (8.4ms RTT), SSH key auth, Ollama serving `qwen2.5-coder:7b` (4.68 GB, Q4_K_M, 32k ctx), model round-trip 11s.

### Design decisions taken before instructing the 7B

I read the full test suite (7 files, the spec) and the PRD before writing any instruction. Two decisions the tests force that the PRD does not state:

D1: Output affine denormalization in both models.
`test_can_overfit_tiny_dataset` demands training MAE < 5 mV after 200 epochs on a 10-sample set (= 200 Adam steps at lr 1e-3). Targets there have mean ~ -80 mV, std ~ 26 mV. Adam's per-step parameter movement is bounded by ~lr regardless of gradient magnitude, so 200 steps moves any parameter ~0.2. A decoder initialized at ~0 output cannot reach -80 mV in that budget. Therefore both `MPNN` and `BaselineMLP` end with `out = raw * 30.0 + (-50.0)` (fixed constants, not learned). This is target denormalization, not an activation, so it stays inside the PRD's "no final activation" for the decoder. Verified compatible with every other test that touches model output (`test_residual_connections_active`, `test_edge_features_affect_output` atol 1e-6, `test_scaling_sensitivity` atol 1e-4, `test_predictions_in_physical_range_after_training` bounds -200/+150).

D2: Trainer stays on CPU.
`test_full_pipeline` and `test_checkpoint_save_and_load` call `model(...)` on CPU tensors *after* `trainer.train(...)` and then `.numpy()`. If the Trainer moved the model to CUDA it would leave it there and every post-training test would fail. The Trainer therefore does not touch device placement. The GPU remains available for real Phase 1 training runs outside the test suite.

Additional constraints extracted from the tests that the PRD does not mention, to be encoded in instructions when the relevant task comes up:
- `BioelectricDataset` must attach `config_id` to each `Data`. `test_no_split_leakage` falls back to `str(i)` when it is absent, which collides across splits and hard-fails.
- Per-split processed-cache filenames, or the four splits overwrite each other's cache.
- Node-feature normalization divides by fixed per-channel maxes `[50,30,20,10,15,30,10,10]` with no clipping; the `exogenous_expression` OOD perturbation legitimately exceeds 1.0 and only train-split normalization is asserted.
- `Trainer` must drive both `MPNN(x, edge_index, edge_attr)` and `BaselineMLP(x)` through one `.train()`.
- `mean` aggregation in the MPNN must survive zero-edge and single-node graphs.

### Task 3.1: package skeleton
- Six empty `__init__.py` files created directly (zero lines of code, no 7B call warranted).
- Status: done.

### Coherence State
Tests passing: 0/0; suite not yet runnable, `torch`/`torch_geometric` installing on the rig.
Current tier: 1 | Scaffold: full

---

## [Session 1] Milestones 3-6: Build

Executor: `qwen2.5-coder:7b`, temperature 0.1, seed 7, num_ctx 16384. One instruction, one file, one test run.

### Task ledger

| # | File | Tier | Scaffold | Result | Tests |
|---|---|---|---|---|---|
| 3.1 | six `__init__.py` | - | - | done (empty files, no 7B call) | - |
| 4.1 | `nexus/model/losses.py` | 1 | full | **pass, 1st attempt** | TestLosses 7/7 |
| 6.1 | `nexus/evaluation/metrics.py` | 1 | full | **pass, 1st attempt** | TestMetrics 10/10 |
| 3.2 | `nexus/data/validation.py` | 1 | full | **pass, 1st attempt** | TestDataValidation 5/5 |
| 5.1 | `nexus/training/config.py` | 1 | full | **pass, 1st attempt** | (used by TestTrainer 10/10) |
| 6.2 | `nexus/evaluation/generalization.py` | 1 | full | **pass, 1st attempt** | TestGeneralizationEval 1/1 |
| 6.3 | `nexus/evaluation/figures.py` | 2 | full | **pass, 1st attempt** | TestFigures 2/2 |
| 4.2 | `nexus/model/baseline.py` | 2 | full | **pass, 1st attempt** | TestBaselineMLP 3/3 |
| 4.3 | `nexus/model/mpnn.py` | 3 | full | **pass, 1st attempt** | TestMPNN 14/14 |
| 5.2 | `nexus/training/trainer.py` | 3 | **partial** | **FAIL** | crashed on test 1 of 10 |
| 5.2r | `nexus/training/trainer.py` | 3 | full (escalated) | **pass, 1st attempt** | TestTrainer 10/10 |
| 3.3 | `nexus/data/config_sampler.py` | 2 | full | **pass, 1st attempt** | TestConfigSampler 9/9 |
| 3.4 | `nexus/data/dataset.py` | 3 | full | **pass, 1st attempt** | TestBioelectricDataset 16/16 |

11 source files. 12 instructions sent. 1 correction round.

### Property 2: Tiered engagement

Promotion history:
- Started Tier 1.
- Harness-forced early promotion to Tier 2 after 5 Tier 1 files written but 0 verified. Reason, recorded because it is a deviation: no test module in this suite can be imported until *every* module it imports exists. `tests/test_evaluation.py` imports metrics (T1), generalization (T1) and figures (T2) at module scope, so the two Tier 1 files were unverifiable until the Tier 2 file existed. The tier gate ("promote after 3 Tier 1 tasks pass") and the import contract deadlock each other. I broke the deadlock on the cheapest Tier 2 file rather than stalling the build. `figures.py` passed first attempt, so the promotion was not premature.
- Tier 2 -> Tier 3 confirmed by `mpnn.py` passing 14/14 on the first attempt. No demotion was ever required.

The 7B showed no tier-correlated failure. Its single failure was at Tier 3, but the cause was scaffold level, not task tier; the same file at the same tier passed on the next attempt with a fuller instruction.

### Property 5: Scaffold degradation: the self-falsification test fired

After 6 verified consecutive first-attempt passes at full scaffold, I promoted to partial scaffold and ran `nexus/training/trainer.py` at the degraded level: file path, class name, one sentence of behavior, the interface surfaces Property 3 allows, and the test class source. No signature, no imports, no line-by-line behavior.

Scaffold degradation test: 7B failed `trainer.py` at partial scaffold. Re-attempting at full scaffold. Previous convergence was scaffold-dependent.

What it produced, from the raw file:
1. `patience -= 1` on a name never assigned -> `UnboundLocalError` on the first test. This is what the suite caught.
2. `save_checkpoint` referenced a bare `history`, a module-level name that does not exist -> would have been a second `NameError`.
3. Called `self.model(batch.x, batch.edge_index, batch.edge_attr)` unconditionally, so `test_baseline_mlp_trains` could not have passed; the MLP takes one argument.
4. `from torch_geometric.data import DataLoader`; that import path was removed in PyG 2.x.
5. `Adam` not `AdamW`; `StepLR` not cosine annealing; never wrote `best.pt`; early-stopping compared against the previous epoch rather than the best epoch.

Five independent defects. Only one was a test-visible crash; the other four were latent and would have surfaced one at a time. The 7B could reproduce a *shape* of training loop from the tests but could not infer the contracts the tests only imply.

Re-issued at full scaffold, the same model produced a correct file in one attempt, 10/10.

Reading: the capability boundary sits between "transcribe a fully specified file" and "infer an unstated specification from its tests". The 7B is reliably on the near side of that line and reliably not on the far side. The nine earlier first-attempt passes were a property of instruction completeness, not of the model having internalized the project. Scaffold tier reset to full, as the rule requires.

### Design decisions I made and the 7B implemented

Recorded because these are mine, not the PRD's, and a reader of the code should know where they came from. D1 (output affine denormalization, `raw * 30.0 - 50.0` in both models) and D2 (Trainer never touches device placement) are described in the Milestone 0 entry above. D1 is directly load-bearing: `test_can_overfit_tiny_dataset` demands < 5 mV training MAE inside 200 Adam steps, and it passed.

Third decision, taken at `trainer.py`: model dispatch by forward-signature arity. The one `Trainer` must drive both `MPNN.forward(x, edge_index, edge_attr)` and `BaselineMLP.forward(x)`, and `test_baseline_mlp_trains` and `test_gnn_beats_mlp_on_coupled_data` both route the MLP through it. `inspect.signature(model.forward).parameters` is checked once in `__init__` rather than try/except per batch, so a genuine `TypeError` inside a model's forward is never silently swallowed as a dispatch miss.

Fourth: the PRD's optional physics auxiliary loss is implemented but not wired into training. `physics_auxiliary_loss` exists and passes its tests. Adding it at ??=0.01 would perturb the tight MAE budget in `test_can_overfit_tiny_dataset` for no test-visible gain. It is available for the real Phase 1 training runs; PRD sec 5.2 marks it optional.

### Known latent issue, not test-visible

`nexus/data/validation.py`, the physical-range check:

    if np.any(finite & (vmem < -120.0) | (vmem > 60.0)):

`&` binds tighter than `|`, so this parses as `(finite & (vmem < -120)) | (vmem > 60)` and the `finite` mask does not guard the upper comparison. Consequence: an `Inf` element reports both "contains Inf values" and "outside physical range", where only the first is meant. No test distinguishes these, all 5 validation tests pass, and no caller is affected. Logged rather than silently fixed, because fixing it is a code change and I do not write code. Worth one 7B correction round in a cleanup pass.

### Coherence State
Tests passing: 87/87 runnable at last per-module count (13 + 24 + 10 + 30 + numerical + integration pending full-suite confirmation).
Last 12 tasks: pass, pass, pass, pass, pass, pass, pass, pass, fail, pass, pass, pass
Current tier: 3 | Scaffold: full (reset after the degradation event)
Regressions: none. No test that passed has since failed.

---

## [Session 1] Milestone 7: Integration: Complete

Cleanup round: re-specified `nexus/data/validation.py` to fix the operator-precedence issue logged
above. The 7B reproduced the file with exactly the one line changed (the only other diffs were
trailing-whitespace normalization). This was not a test failure, so the Property 1 correction
template did not apply; there was no failing output to hand back. It was issued as a fresh
full-scaffold specification of the same file, with the current file supplied verbatim as fact.

Full suite, final run:

    collected 92 items / 5 deselected / 87 selected

    tests\test_data_pipeline.py ..............................  [ 34%]
    tests\test_evaluation.py    .............                   [ 49%]
    tests\test_integration.py   ...                             [ 52%]
    tests\test_model.py         ........................        [ 80%]
    tests\test_numerical.py     .......                         [ 88%]
    tests\test_training.py      ..........                      [100%]

    ================ 87 passed, 5 deselected in 180.82s (0:03:00) =================

The 5 deselected are `@pytest.mark.betse`, which require BETSE installed. Excluding them is the
stated success criterion.

The behavioral tests are the meaningful ones, and they all hold on trained models:
- `test_gnn_beats_mlp_on_coupled_data`: the MPNN beats the per-cell MLP on neighbour-dependent
  targets, so the graph structure is genuinely contributing and message passing is not decorative.
- `test_can_overfit_tiny_dataset`: training MAE under 5 mV inside 200 optimizer steps.
- `test_identical_cells_identical_vmem`: graph automorphism respected to 1e-5.
- `test_permutation_equivariance`: node relabelling permutes the output identically.
- `test_higher_coupling_reduces_vmem_variance`: the model distinguishes coupling regimes.
- `test_predictions_in_physical_range_after_training`: no runaway extrapolation.

### Final Coherence State
Tests passing: 87/87 runnable (100%), 5 deselected (BETSE).
Files: 11 source modules + 6 package `__init__.py`, all written by qwen2.5-coder:7b.
Instructions sent: 13. Correction rounds: 1 (`trainer.py`, after the deliberate partial-scaffold
degradation test) + 1 cleanup re-spec (`validation.py`, no test failure involved).
First-attempt pass rate at full scaffold: 12/12. At partial scaffold: 0/1.
Current tier: 3 | Scaffold: full
Regressions across the whole session: none.

### What is NOT done, and why

Milestone 1 (BETSE integration) and Milestone 2 (data generation) are untouched. CLAUDE.md
explicitly defers Milestone 1: "BETSE installation is a risk. Start with Milestone 3"; and the
5 deselected tests are exactly its acceptance criteria. `nexus/data/betse_generator.py` does not
exist. The core pipeline it was deferred behind now works end to end, so BETSE is the next thing
to attempt.

Also absent, all of them outside the test suite's acceptance criteria: the training entry-point
scripts under `scripts/`, the YAML configs under `configs/`, wandb/TensorBoard logging (PRD sec 6.3),
the speed benchmark (PRD sec 7.3), and the ablation studies (PRD sec 7.4). The physics auxiliary loss is
implemented and tested but deliberately not wired into the training loop.

### Note on the executor

qwen2.5-coder:7b wrote every line of source in this build. It never once produced a design; it
produced transcriptions of designs. The single failure came the one time it was asked to infer a
specification rather than implement one. Median response time roughly 15-30 seconds per file at
temperature 0.1. The bottleneck in this loop was never the 7B's speed; it was the precision of the
instruction handed to it.

---

## [Session 2] Milestone 1: BETSE Integration

### The install risk did not materialize

The premise going in was that BETSE was last maintained around 2021 and would fight Python 3.12.
That is wrong, and worth correcting in the record:

- BETSE 1.5.0, uploaded to PyPI 2025-04-15. (1.4.1 was 2024-09-24.) Actively maintained.
- `requires_python: >=3.11`, with trove classifiers for 3.10 through 3.13.
- Declared deps: `beartype>=0.18`, `dill>=0.2.3`, `matplotlib>=3.9`, `numpy>=2.0`, `pillow>=5.3`,
  `ruamel-yaml>=0.15.24`, `scipy>=1.14`. The rig already satisfied every one.
- `pip install betse` succeeded first try, pulling only 4 new packages. `torch`, `torch_geometric`,
  `numpy` and `scipy` were all left untouched; the 87 non-BETSE tests still pass.

No fallback finite-difference solver is needed. The PRD's contingency stays unused.

### Infrastructure findings

A real BETSE bug on Windows. `betse config <path>` validates that its argument is not a bare
basename, and its check recognizes only the backslash as a directory separator. Given a forward
slash it raises

    BetsePathnameException: Pathname "sim/config.yaml" contains no directory separators
    (i.e., '\' characters).

but only *after* writing the YAML, so it exits 1 having produced 1 file, where the backslash
form exits 0 and produces 136, including `sim\extra_configs\expression_data.yaml` and the
`sim\geo\` image assets. The `seed` stage then dies reading the missing expression_data.yaml.

This cost one correction round, and the fault was mine, not the 7B's: my instruction told it to
pass the literal string `"sim/config.yaml"`. The corrected instruction specifies `os.path.join`,
which yields the backslash form on this platform. Recording it because it is exactly the class of
error the director is supposed to absorb; the 7B transcribed my spec faithfully and my spec was
wrong.

Default config is unusable as shipped. Two changes are required before any run:
- `general options -> simulate extracellular spaces` must be `False`. The extracellular solver
  aborts inside `sim_toolbox.nernst_planck_flux` at the world sizes used here.
- Every scheduled intervention must be switched off (`event happens: False`, walked recursively).
  The shipped config fires a *cutting event* mid-simulation that removes cells.
- `ion profile` is raised from `basic` (Na, K, M, P) to `mammal` so Cl and Ca exist, since the PRD
  parameter space includes Cl and Ca channels.
- There is no `plot while solving` key anywhere in the config. An earlier probe of mine set one and
  ruamel silently created it; the verification pass caught that it was junk before it reached an
  instruction.

Verified output API (BETSE gives no documentation for this; established by introspection):

| Quantity | Source | Units |
|---|---|---|
| steady-state Vmem | `sim.vm_ave` | volts -> x1000 for mV |
| cell positions | `cells.cell_centres` | metres -> x1e6 for um |
| GJ topology | `cells.cell_nn_i`, shape (n_membranes, 2) | cell index pairs |
| GJ open fraction | `sim.gjopen` | same length as `cell_nn_i` |
| Vmem time series | `sim.vm_ave_time` | list of per-cell arrays |

The sim pickle is a 3-list `[Simulator, Cells, Parameters]`; the world pickle is a 2-list. Rows of
`cell_nn_i` whose two entries are equal are boundary membranes facing no neighbour and must be
masked out, or the graph acquires self-loops that `test_no_self_loops` would reject downstream.

Cell count is emergent, not settable. BETSE has no "number of cells" parameter; the count falls
out of meshing a square world of a given size with a given cell radius. Calibrated over 7 seed-only
runs:

| world um | radius um | n_cells |
|---|---|---|
| 80 | 5 | 42 |
| 120 | 5 | 109 |
| 200 | 5 | 320 |
| 300 | 5 | 711 |
| 150 | 10 | 31 |
| 250 | 10 | 116 |
| 200 | 15 | 19 |

`world_um ~ ?? * radius_um * sqrtn_cells` with ?? ~ 2.28 across the 100-700 cell band that matters
(?? drifts up at very low counts). Accurate to roughly +/-10% on world size. `BETSEGenerator`
therefore treats `n_cells` as a *target* and reports the actual count in the returned record.

### Task ledger, session 2

| # | File | Tier | Scaffold | Result |
|---|---|---|---|---|
| 1.1 | `nexus/data/betse_config.py` | 2 | full | **pass, 1st attempt** |
| 1.2 | `nexus/data/betse_generator.py` | 3 | full | fail; 4/5, caused by my forward-slash spec error |
| 1.2r | `nexus/data/betse_generator.py` | 3 | full | 3/5 pass; 2 remaining are physics calibration, not code |

### Known defect in the current betse_generator.py

The `finally:` block the 7B produced is inverted:

    finally:
        if not self.keep_files:
            work = None
        if work is not None and not self.keep_files:
            shutil.rmtree(work, ignore_errors=True)

The first branch nulls the variable that the second branch tests, so the temporary directory is
never removed. No test detects this. It does not matter for a 5-test run; it matters a great
deal for Milestone 2, where 12,000 simulations would each strand a 136-file scaffold in the temp
directory. Queued for a correction round.

### Recalibration: finding the right BETSE knob

The two remaining failures were both mine; wrong physical mapping, not wrong code. I measured the
response surface rather than guessing again.

Sweep 1, the two passive membrane constants. Median Vmem in mV:

| Dm_Na \ Dm_K | 5.0e-19 | 1.0e-18 | 2.0e-18 | 4.0e-18 |
|---|---|---|---|---|
| **5.0e-19** | -75.9 | -76.3 | -77.0 | -78.2 |
| **1.0e-18** | -60.6 | -61.4 | -62.9 | -65.2 |
| **2.0e-18** | -43.4 | -44.6 | -44.9 | -36.5 |
| **4.0e-18** | -18.6 | -19.1 | -19.9 | -20.5 |

Read down a column and Vmem swings 57 mV. Read across a row and it moves 1-3 mV. `Dm_K` is
almost inert. That single fact explains both failures at once: my original mapping put Kir and
K_leak onto `Dm_K`, so a full Kir blockade moved Vmem by 0.20 mV, and it put Nav onto `Dm_Na` with a
0.1xbase floor, which at Nav=5 pushed Vmem to -117 mV.

Sweep 2, the real potassium lever. BETSE keeps voltage-gated channels in a separate list at
`general network -> channels`, shipped with three entries (`Nav`/Nav1p3, `Kv`/Kv1p5, `K_Leak`/KLeak),
each with a `max Dm`. BETSE also provides a `Kir2p1` inward rectifier. Repurposing the `Kv` slot:

| Kir2p1 `max Dm` | median Vmem |
|---|---|
| 0.0 | -44.61 |
| 1.0e-17 | -46.67 |
| 1.0e-16 | -63.37 |
| 1.0e-15 | -85.17 |

A 40 mV span, in the physiologically correct direction. That is the lever.

Sweep 3, sodium through the channel list. Raising the `Nav` entry's `max Dm` from 0.0 to 1.0e-15
moved Vmem from -63.38 to -63.30 mV; 0.08 mV, nothing. This is correct physics, not a bug: a
voltage-gated sodium channel is shut at a resting potential of -63 mV. Nav therefore stays on the
passive `Dm_Na` constant, and the `Nav` channel entry is pinned to 0.0.

A further detail that mattered: the shipped `Nav` and `Kv` entries carry `init active: false`, so
they are inert during the `init` stage where the steady state is actually reached. All three
entries are forced to `init active: true`.

Final mapping, verified against the exact test configurations before it was ever written into an
instruction:

| PRD channel | BETSE target | Range |
|---|---|---|
| Nav | `Dm_Na` | `1e-18 * (1.0 + 3.0*f)` |
| Kir | `Kir2p1` channel `max Dm` | `3e-16 * f` |
| K_leak | `K_Leak` channel `max Dm` | `2e-17 * f` |
| Ca | `Dm_Ca` | `1e-18 * (0.1 + 1.9*f)` |
| Cl | `Dm_Cl` | `1e-18 * (0.1 + 1.9*f)` |
| NaKATP | `alpha_NaK` | `1e-7 * (0.5 + 1.0*f)` |
| HKATP, VATP | unmapped | - |
| gj_conductance | `gap junction surface area` | `1e-9 -> 1e-7` |

Pre-flight results on the three test configurations, before instructing the 7B:

    C single_sim   median  -71.14 mV   (needs within [-120, 60])
    A plausible    median  -82.07 mV   (needs -100 < x < 20)
    B base         mean    -77.60 mV
    B Kir=0        mean    -56.67 mV   -> blockade delta 20.93 mV (needs > 1.0)

Every one clears with margin. Only then did the corrected instruction go out.

HKATP and VATP are accepted and discarded. BETSE's configuration exposes `alpha_NaK` and
`alpha_Ca` and no H+/K+ or V-ATPase pump. Mapping VATP (a proton pump) onto `alpha_Ca` (a calcium
pump) would be inventing physics, so both are left unmapped. Consequence for Milestone 2: two of the
eight input channels will be pure noise to the model, uncorrelated with the target. Either
`ConfigSampler` should hold them constant for BETSE-backed runs, or the feature vector should drop
to 6 channels. This is an open decision and must be settled before generating the real dataset.

### Infrastructure finding: SSH dies under BETSE load

Long BETSE runs saturate the CPU and Windows `sshd` stops responding; sessions carrying test output
drop with `Read from remote host: Operation timed out` partway through. Three separate runs were lost
this way, and a parallel polling session made it worse by consuming the concurrent-session budget.

Working pattern, needed again for Milestone 2's bulk generation:

    Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd /c ...'}

which fully detaches the process from the SSH session, writing to a log file on the rig that short
follow-up calls read. `Start-Process` is *not* sufficient; its child dies with the parent session.

### Task ledger, session 2 (final)

| # | File | Tier | Scaffold | Result |
|---|---|---|---|---|
| 1.1 | `betse_config.py` | 2 | full | **pass, 1st attempt** |
| 1.2 | `betse_generator.py` | 3 | full | fail; my forward-slash spec error |
| 1.2r | `betse_generator.py` | 3 | full | 3/5; my physics mapping error |
| 1.1r | `betse_config.py` | 2 | full | **pass, 1st attempt** (remapped) |
| 1.2r2 | `betse_generator.py` | 3 | full | **5/5 pass** |
| 1.2r3 | `betse_generator.py` | 3 | full | **pass**; added missing `except` |

Four corrections on `betse_generator.py`. Three of the four were caused by my instructions, not by
the 7B: the forward-slash path, the wrong physics mapping, and a spec the 7B followed while
dropping one clause. Only the dropped `except Exception` and the inverted `finally` were the 7B's own
errors, and both were in the boilerplate around the logic rather than in the logic itself.

### Final Coherence State
Tests passing: 92/92 (100%), including all 5 `@pytest.mark.betse` tests. Nothing deselected.
Leaked temp directories after a full run: 0.
Files: 14 source modules + 6 package `__init__.py`, every line written by qwen2.5-coder:7b.
Current tier: 3 | Scaffold: full
Regressions: none. No test that passed has since failed.

---

## [Session 2] Decision: HKATP and VATP held at zero

The open question from the BETSE integration (two of the eight input channels have no BETSE
equivalent and would be pure noise) is now settled. Hold them constant at zero. Keep 8 channels.

Decision and rationale are the user's, recorded here because a reader of the code will otherwise
find two permanently-zero feature columns and assume it is a bug:

1. The test suite hardcodes 8. `N_CHANNELS = 8` in conftest.py, in every model constructor call,
   in every fixture, and in the dataset normalization constants. Dropping to 6 cascades through the
   whole verified harness for no scientific gain.
2. Phase 2 transfer. Phase 2's encoder takes transcriptomic input. An encoder trained on a
   6-wide input does not match Phase 2's input space and its pretrained weights do not transfer --
   which is the entire reason Phase 1 exists. Training on 8 dimensions where 2 carry no information
   leaves near-zero weights on those dimensions, and that is the *correct* initialization for
   channels whose Phase 2 predictive value is unknown.
3. Cost. One conditional in one file.

### Implementation

`ConfigSampler.__init__` gains `zero_unmapped_channels=True`. Inside the row loop, columns 6 and 7
(`UNMAPPED_CHANNEL_INDICES`) are set to 0.0 immediately after the density row is tiled and before
the perturbation block, so a perturbation can never be silently erased afterwards. The flag defaults
to the BETSE-backed behaviour, so the tests, which construct `ConfigSampler()` and
`ConfigSampler(seed=42)`; get the zeroed columns with no test changes. Passing False restores full
8-channel sampling.

### One thing the specification did not cover, added on my own judgement

The three perturbation branches that target a channel drew their index with
`self.rng.integers(0, 8)`. With columns 6 and 7 pinned to zero, roughly 25% of
`channel_blockade`, `exogenous_expression` and `spatial_gradient` configurations would have selected
a column that is already zero and done nothing at all; while still being written to disk labelled
`is_perturbation: True` and routed into the `test_ood` split. A quarter of the out-of-distribution
generalization set would have been unperturbed baselines masquerading as perturbations, which would
have made OOD performance look better than it is.

The draw is now `self.rng.integers(0, n_perturbable)` with `n_perturbable = 6` when zeroing is on.

### Verified empirically, not just by the test suite

    baseline per-column max over 100 configs:
       [49.71 29.71 19.84  9.96 14.98 29.84  0.    0.  ]
      cols 6,7 all zero: True
      cols 0-5 nonzero somewhere: True
    channel_blockade         touched cols=[0,1,2,3,4,5]  no-op configs=0/60
    exogenous_expression     touched cols=[0,1,2,3,4,5]  no-op configs=0/60
    spatial_gradient         touched cols=[0,1,2,3,4,5]  no-op configs=0/60
    escape hatch (zero_unmapped_channels=False): cols 6,7 max = [9.87 9.91]
    reproducible: True

Columns 0-5 still span their full PRD ranges, so zeroing 6 and 7 cost no coverage elsewhere.

Test coverage note. `test_channel_blockade_zeroes_channel` asserts only that *some* column is
entirely zero. Columns 6 and 7 now always satisfy that, so the test would pass even if the blockade
did nothing. It is no longer evidence that blockade works. The measurement above is what establishes
that, and it is the reason the check was run at all rather than trusting a green suite.

### Coherence State
Tests passing: 92/92 (100%), full suite including BETSE, 461s.
Regressions from this change: none.
Current tier: 3 | Scaffold: full

---

## [Session 3] Milestone 2: Data Generation

### Files built

| # | File | Tier | Scaffold | Result |
|---|---|---|---|---|
| 2.0 | `betse_generator.py` (+convergence, +timeseries) | 3 | full | fail; 7B deleted the extraction block |
| 2.0r | `betse_generator.py` | 3 | full | **5/5 BETSE tests pass** |
| 2.1 | `nexus/utils/io.py` | 1 | full | **pass, 1st attempt** |
| 2.2 | `scripts/generate_dataset.py` | 3 | full | crash on 10th completion |
| 2.2r | `scripts/generate_dataset.py` | 3 | full | **24/24 sims, resume verified** |
| 2.3 | `scripts/finalize_dataset.py` | 2 | full | **pass, 1st attempt** |
| 2.4 | `betse_generator.py` (+per-run log file) | 3 | full | **pass, 1st attempt** |

A new 7B failure mode, and the most instructive one yet. Asked to *insert* a block into a long
file, it deleted the six adjacent lines that assigned `positions`, `nn`, `keep`, `edge_index`,
`gjopen` and `conductances`. The returned dict still referenced three of them, so every call raised
NameError, the `except` swallowed it, and 4 of 5 tests failed with "returned None". My instruction
said "after the line that computes `vmem` add these lines", and the 7B read that as *replace what
follows*. The fix was to hand it the complete twenty-two line replacement block with the note that
"every line of the original nine appears again below, in the same order". It then reproduced all of
it. Lesson: never phrase an edit to a long file as an insertion. State the entire replacement
region verbatim. The same phrasing worked fine on short files.

Second 7B error, caught by the pilot: `mean_wall_clock = total_wall_clock_s / (completed - done)`,
where `done` is a *set* of already-finished config ids. `int - set` raises TypeError. It crashed on
the 10th completion, after 10 simulations had already been written, which is exactly why the pilot
was sized to reach that code path rather than merely to check that something ran.

### Measurements taken before committing to a long run

Throughput is not what per-simulation latency suggests. Per-sim wall clock at 8 workers was
roughly double that at 6, which looks like saturation. Measured properly on *identical* work
(same 24 configurations, fresh staging directory each time):

| workers | elapsed | throughput | peak RAM |
|---|---|---|---|
| 6 | 323.1s | 267.4 sims/hr | - |
| 12 | 245.4s | 352.1 sims/hr | - |
| 18 | 245.3s | 352.2 sims/hr | 17.6 GB |

Latency rose but throughput rose with it, and then flatly plateaued: 18 workers bought nothing
over 12 while using more memory. 12 workers chosen. Had I trusted the latency reading I would
have run this at 6 workers and taken 52 hours instead of 36.

Caveat on the measurement, recorded honestly: 24 tasks over 12 or 18 workers is only two dispatch
rounds, so both numbers are quantised the same way and the true 18-worker figure could differ. The
plateau plus the memory cost is enough to prefer 12; it is not a precise scaling curve.

Pilot quality, 24 simulations, 0 failures:
- `converged_fraction` = 1.0 on every run; every cell settled.
- `max_dvmem_mv` ~ 1e-5 mV, four orders of magnitude inside the PRD's 0.1 mV steady-state criterion.
- Cell counts track the ?? = 2.28 calibration well at scale (492->490, 373->367, 275->261) and
  undershoot at the low end (53->41, 62->45), as expected from the calibration data.
- 24.1 KB mean per .npz, so the full dataset is well under 1 GB.

Two infrastructure fixes. Every worker's `betse` was writing to one shared default log file and
rotating it under the others, producing a stream of `LogHandlerFileRotateSafe.emit` tracebacks. Each
run now gets `--log-file` inside its own temporary directory. And the rig is a laptop: sleep and
hibernate timeouts were set to 0 before launch, since a 36-hour run would otherwise be suspended.

### Run launched

    python scripts/generate_dataset.py --n-baseline 11500 --n-per-perturbation 575 \
        --jobs 12 --timeout 1200 --seed 42 --timeseries-count 1000 --out data/synthetic/_staged

13,800 attempts for 12,000 required simulations; the PRD's 15% oversample against an anticipated
5-15% BETSE failure rate. Observed failure rate so far is 0 of 120, so most of that margin will
likely go unused; the spec was followed anyway rather than trimmed on thin evidence, since
perturbation configurations (gj_conductance = 0, densities at 4x the training maximum) are the ones
most likely to stress the solver and none had been sampled at the time of the decision.

Launched detached via `Win32_Process.Create`, so it survives the SSH disconnections that BETSE load
reliably causes. Progress and every outcome are flushed to `data/synthetic/_staged/results.csv`
after each completion, so an interruption at any point loses nothing: re-invoking the script reads
that file and skips what is done.

First reading after 4 minutes: 20/13800 complete, 20 ok, mean 113.9 s/sim, ETA 36.3 hours.

`scripts/finalize_dataset.py` runs after generation. It sorts the successful runs by config id,
takes 8000/1000/1000 for train/val/test_id and caps test_ood at 500 per perturbation type, copies
(not moves, by default) each file into its split directory and writes `manifest.csv`.

### Coherence State
Tests passing: 92/92. Generation run in progress.
Current tier: 3 | Scaffold: full

### Hands-off completion: on-rig watcher

The run is unattended. Polling it from the Mac would be self-defeating; sustained BETSE load is
exactly what kills SSH sessions here, and every poll is a session that can die mid-read. The
completion signal therefore comes from the rig itself, not from the director.

`~/nexus-ops/watcher.py` (director operations tooling, deliberately outside `nexus-phase1/` because
it is not part of the deliverable) runs detached and:

1. Polls the local `gen_full.log` once a minute for the `RUN_COMPLETE` marker; a local file read,
   no network, no SSH.
2. Also detects an abnormal stop: log present but no `python.exe` processes left. That case reports
   "STOPPED EARLY" rather than pretending success.
3. Runs `scripts/finalize_dataset.py` to build the splits and the manifest.
4. Sends one ntfy notification carrying real numbers: ok/total and success rate, a breakdown of
   every failure status, the four split sizes, mean `converged_fraction` with a count of runs below
   the PRD's 0.95 threshold, total CPU hours, and the finalize return code.
5. Escalates the notification to high priority with a warning tag if the run stopped early, finalize
   returned non-zero, or any split came up short of its target.

Its notification path was tested end to end before arming it, so the one signal that matters is
known to work rather than assumed to.

### finalize_dataset.py had a latent crash, found by testing it early

Run against the partial staging directory while generation was still going, it died immediately:

    key=lambda x: int(x["config_id"])
    ValueError: invalid literal for int() with base 10: 'base_000011'

Config ids are strings such as `base_000011` and `gj_blockade_000108`; my instruction said "sorted by
the config_id field" and the 7B added an int() cast. Sorting the padded strings is what the split
assignment actually wants.

Worth recording *why* this was caught: the script was never going to run until the generation
finished, roughly 36 hours away, and it would then have crashed instantly and produced no dataset.
Testing it against the 78 records that already existed cost about a minute and moved that failure
from hour 36 to hour 0. The corrected version was re-tested the same way: 86 successful rows, all
86 assigned to train, val and test_id correctly empty because train's target of 8000 is not yet
filled, test_ood empty because the work list runs every baseline before any perturbation, and all
four shortfall warnings printed accurately. The manifest wrote forward-slashed relative paths as
specified.

### Status at handoff

    completed 90/13800  ok=90  mean=123.1s  eta=39.07h

90 of 90 successful. Nothing further from the director until the rig reports in.

### Coherence State
Tests passing: 92/92. Generation running unattended, watcher armed.
Current tier: 3 | Scaffold: full

---

## [Session 3] Finding: the target is not a steady state, and cannot be made one

Mid-run, while watching the generation job, I questioned whether the simulations were running
longer than necessary. Every record reported `max_dvmem_mv` around 1e-5 mV, four orders of magnitude
inside the PRD sec 4.1.4 criterion of 0.1 mV, which looked like an obvious over-simulation and a free
2-3x speedup. Two measurements later that hypothesis was dead and had been replaced by a real
data-quality finding pointing the opposite way.

### First correction: my convergence metric was measuring the wrong phase

BETSE's default time settings, which the generator never overrode:

    init time settings:  time step 0.01    total time 5.0     sampling rate 1.0
    sim time settings:   time step 0.0001  total time 0.035   sampling rate 0.001

`init` runs 5 seconds of simulated time and is what equilibrates the tissue. `sim` runs
35 milliseconds at fine resolution on an already-equilibrated state. `BETSEGenerator` computes
`converged_fraction` and `max_dvmem_mv` from `sim.vm_ave_time`, which is the *sim* phase, so it was
measuring drift over 1 ms at the end of a window that starts from equilibrium. Of course it was
1e-5 mV. It proves the sim phase is quiet. It says nothing about whether init converged. I had
been quoting it as convergence evidence for two sessions.

### Second measurement: skipping `sim` is safe and nearly worthless

Four already-computed configurations re-run as seed+init only, compared cell-by-cell against their
stored production values:

| config | production | init-only | max diff | speedup |
|---|---|---|---|---|
| base_000000 | -78.12 | -78.12 | 0.0009 mV | 1.54x |
| base_000001 | -12.14 | -12.14 | 0.0027 mV | 1.50x |
| base_000002 | -70.54 | -70.55 | 0.0022 mV | 1.57x |
| base_000003 | -80.19 | -80.19 | 0.0003 mV | 1.63x |

The 35 ms sim phase changes Vmem by under 0.003 mV. It contributes nothing but costs about a third
of the runtime. But the same test showed extending `init` from 5 s to 15 s moves the answer --
`base_000001` shifted 0.628 mV, so the question was no longer "can we go faster" but "are we
stopping too early".

### Third measurement: there is no steady state to converge to

Three configurations re-simulated with `init` extended to 40 s, sampled every 2 s. Per-interval
max |dVmem|, and the fraction of cells meeting the 0.1 mV criterion:

    base_000001 (n=261)
      t        2s    6s   12s   20s   28s   36s
      maxd   0.195 0.154 0.175 0.173 0.162 0.154
      frac<.1 0.00  0.00  0.00  0.00  0.00  0.00
      mean Vmem drifted 2.00 mV from t=5s, still moving at t=36s

    base_000002 (n=490)  frac<.1 oscillates 0.89-0.96, never holds above 0.95, drift 1.03 mV
    base_000000 (n=85)
      t        2s    4s    6s   12s   20s   28s   36s
      maxd   0.144 0.054 0.042 0.061 0.093 0.105 0.101

The drift does not decay. In `base_000000` it reaches a minimum at 4-6 s and then rises again.
That non-monotonic shape is the whole finding: it is not a single relaxation approaching a limit, it
is two superposed processes.

1. A fast electrical transient; membrane charging and current redistribution through gap
   junctions, complete by roughly 4-6 s. Electrical time constants here are milliseconds to seconds.
   This is the bioelectric physics the model is meant to learn.
2. A slow secular concentration drift; the Na+/K+-ATPase pumps continuously, and for
   arbitrarily sampled channel densities the pump-and-leak system does not balance. Intracellular
   concentrations creep and Vmem follows. `base_000001` never settles because it is a depolarised,
   high-sodium-permeability configuration with heavy pump turnover.

For randomly sampled channel densities there is generally no reachable steady state. Simulating
longer integrates more concentration drift rather than converging. PRD sec 4.1.4's convergence
criterion is unsatisfiable as written for this parameter space. That is a property of the ground
truth under random sampling, not a defect in the pipeline.

### Decision: continue the run, fix the description rather than the data

The restart premise failed. The "corrected" dataset a restart would have produced does not exist:
40 s of init costs 8x and still leaves `frac<.1` at 0.00 for `base_000001`. Restarting for skip-sim
alone was close to a wash; about 1.55x on the remaining 73%, roughly 25 h to 22 h, against
discarding 9 hours already banked, plus the risk of disturbing a healthy run.

And the bias was not the kind I had described. Every existing sample was produced by the identical
protocol at the identical simulated time, so `config -> Vmem at 5 s` is a deterministic reproducible
function that a model can learn cleanly. What is not true is calling that quantity a steady state.

Actions taken, none of which touch the running job:

- Wrote `data/synthetic/DATASET_CARD.md` recording the real definition of the target, the three
  trajectories above, the two-process explanation, and every other known limitation.
- The npz key `vmem_steady_state` is retained. Renaming the wire format would break the test
  suite, which hardcodes that name in `conftest.py` and `test_data_pipeline.py`, and Rule 9 makes
  the tests non-negotiable. It would also desynchronise the code from thousands of files already
  written. The rename is semantic: everything descriptive now says "Vmem after 5 s
  equilibration".
- `betse_generator.py` was deliberately not edited. Loky recycles worker processes, and a
  recycled worker re-importing a changed module mid-run would produce exactly the inhomogeneous
  dataset the decision was meant to avoid.
- Flagged in the card that `converged_fraction` and `max_dvmem_mv` must not be cited as convergence
  evidence, since they describe the sim phase only.

### What I got wrong, recorded plainly

I quoted `max_dvmem ~1e-5 mV` as evidence of excellent convergence in two separate reports before
checking which phase produced it. The number was real and the inference was wrong. I also never
profiled the simulator's time settings before committing to a 13,800-simulation run; worker-count
scaling was measured carefully, simulated duration was not questioned at all. Both were caught only
because the run was long enough to invite a second look.

### Coherence State
Tests passing: 92/92. Generation continuing unchanged, ~3900/13800.
Target redefined in documentation; no code changed.
Current tier: 3 | Scaffold: full

---

## [Session 3] Fixing the out-of-distribution split

Asked what failures I was watching for, I traced the status taxonomy and noticed, while reasoning
about it, that `BETSEGenerator` reduces the (n_cells, 8) density array to 8 column means. Measured
immediately:

    spatial_gradient      raw per-cell std 8.72  -> stored std 0.00
    exogenous_expression  raw per-cell std 45.04 -> stored std 0.00
    channel_blockade      uniform by construction -> survives
    gj_blockade           does not touch densities -> survives

Two of the four out-of-distribution types were being stored as uniform tissues. `test_ood` would
have contained 1,000 samples that were baselines with a shifted mean, and generalization scores
would have read better than the truth. Separately `channel_densities_to_betse` clamped at 1.0, so a
4x overexpression was simulated as 1x while the stored feature said 4x.

All 10,551 baselines were unaffected; `ConfigSampler` tiles one row for baselines, so the mean
reduction is lossless there, and sampled densities never exceed their maxima so nothing clamped. And
no perturbation record existed yet: they start at simulation 11,500.

### Two infrastructure obstacles, both real

The 7B could not run at all while the job ran. Ollama returned
`timed out waiting for llama-server to start`. The director cannot write code, so this was a hard
block: the fix could not be written until the run stopped. The run was paused (10,551 npz against
10,551 csv rows, exactly consistent, nothing lost).

Windows ephemeral port exhaustion. With the machine idle the failure persisted, now as
`dial tcp 127.0.0.1:49152: Only one usage of each socket address is normally permitted`. Measured:

    dynamic port range   49152-65535 = 16,384 ports
    TimeWait             25,010
    ...to remote port 49672  12,982

25,010 stuck sockets against a 16,384-port range, and they did not drain; 25010, 25010, 25011 over
150 seconds. The ~13,000 to one local port matches loky's socket-based worker IPC on Windows, one
per dispatched task across ~13,800 tasks. The generation run exhausts the ephemeral port pool, and
would eventually have broken itself, not just Ollama. Fixed by widening the range to 20000-65535
(`netsh int ipv4 set dynamicport tcp start=20000 num=45535`; revert with `start=49152 num=16384`).
This must be set before any future full run.

### The fix

BETSE exposes five tissue-profile picker types, one of which is `INDICES`, taking an explicit list
of cell indices with its own diffusion constants; and the configuration states outright that tissue
profiles may be altered after the `seed` stage. That resolves the ordering problem, since cell
indices do not exist until the mesh is meshed.

`BETSEGenerator` now runs `seed`, reads the mesh, resamples densities onto real cell positions by
spatial rank, groups cells into up to 8 bands by their most variable channel, appends one tissue
profile per band, then runs `init` and `sim`, and stores the per-cell densities actually simulated.

Developed and tested entirely in a separate `~/nexus-dev` tree so the paused production tree was
never touched until the suite was green.

### Two failures found by testing, both instructive

`KeyError: 'Spot'`. The first version deleted the shipped tissue profiles before appending its
own. BETSE's `general network` section refers to the profile named `Spot` by name, so the reference
dangled and `init` aborted. The configuration states that later profiles override earlier ones for
the same cell, so appending rather than replacing keeps `Spot` resolvable while the new profiles
still win. One line removed.

A fix that was worse than the bug. The corrected version produced a spatial_gradient on NaKATP:
the gradient stored faithfully (corr(x, density) = 0.995) but Vmem barely moved, 0.351 mV, and that
was mesh boundary variation rather than a response. A tissue profile carries only Dm_Na, Dm_K, Dm_Cl
and Dm_Ca. Kir and K_leak drive globally-scoped channel entries; NaKATP drives a global internal
parameter. Only Nav, Ca and Cl can vary per cell.

Storing an input that varies while its target does not respond is worse than storing a uniform
one; it would teach the model that such gradients have no effect. `ConfigSampler` now restricts
spatial_gradient and exogenous_expression to `SPATIAL_CHANNEL_INDICES = (0, 3, 4)`. That is a
deliberate narrowing of PRD sec 4.1.1, recorded in the dataset card, taken because the alternative was
letting half of those perturbations be silently inert.

### Verification

    spatial_gradient  (Cl)   corr(x,density) 0.995   corr(density,Vmem) -0.736
    spatial_gradient  (Nav)  corr(x,density) 0.997   corr(density,Vmem)  0.997   Vmem -80.18..-51.95 mV
    exogenous_expr    (Cl)   corr(x,density) -0.744  corr(density,Vmem) -0.813
    baseline base_000000     max |dVmem| vs the pre-change result = 0.000e+00 mV   BIT-IDENTICAL

Full suite in the dev tree: 92 passed in 357s. Only then was it deployed and the run resumed:
`Total configurations to run: 3249, Already done: 10551, Jobs: 12`.

### Task ledger

| # | File | Result |
|---|---|---|
| 3.1 | `betse_config.py` (+FRAC_MAX, +group_cells_by_density) | pass, 1st attempt |
| 3.2 | `betse_config.py` (+resample_densities_to_mesh) | pass, 1st attempt |
| 3.3 | `betse_generator.py` (seed/profiles/init split) | KeyError 'Spot'; replaced profiles instead of appending |
| 3.4 | `betse_generator.py` | pass |
| 3.5 | `config_sampler.py` (spatial channel restriction) | pass, 1st attempt |

### Coherence State
Tests passing: 92/92 (dev tree, deployed to production).
Generation resumed: 10,551 done, 3,249 remaining.
Current tier: 3 | Scaffold: full

---

## [Session 3] Milestone 2 COMPLETE

    Summary: Total completed: 3249, Successes: 3249, Failures: 0
    RUN_COMPLETE

Across the whole milestone: 13,800 simulations attempted, 13,800 succeeded, zero failures.
The PRD anticipated a 5-15% failure rate (sec 11.4) and oversampled 15% to compensate. None of that
margin was needed; 1,800 surplus records sit unused in `_staged/`.

### Final composition, verified

    train    8000    baseline
    val      1000    baseline
    test_id  1000    baseline
    test_ood 2000    500 each of the four perturbation types

    manifest rows 12000, duplicate config_ids across splits: 0
    train all baseline: True, test_ood all perturbation: True
    n_cells 40 / 150 / 490 (min / median / max)
    every sampled file carries all eight required keys, Vmem within [-120, 60] mV
    0.11 GB across 12000 files

### The perturbation phase, and two predictions that were wrong

`gj_blockade` was the feared case and it never failed: 575/575. The reasoning behind the worry
was sound; zero gap-junction conductance should make the gap-junction Laplacian singular, and BETSE
precomputes a dense inverse of it. The prediction was wrong because
`gj_conductance_to_surface_area` floors the surface area at `GJ_SURFACE_CLOSED = 1e-9` rather than
true zero. Cells become nearly isolated but the matrix stays invertible. That floor was chosen for
physical plausibility and incidentally protected the numerics.

The per-cell tissue profile path, written this session and never run in production, worked
first time. Verified on real output rather than by absence of crashes: 8 of 8 sampled
`exogenous_expression` records showed `corr(x, density) ~ -0.75` with the target responding,
`corr(density, Vmem)` from -0.19 to -0.91. Sampled `spatial_gradient` records showed
`corr(x, density) = 0.997` spanning the full channel range.

### The finding that matters for evaluation

Grouping sampled gradients by perturbed channel:

    Nav   mean |corr(density,Vmem)| 0.965   mean Vmem sd 10.58 mV
    Cl    mean |corr(density,Vmem)| 0.533   mean Vmem sd  0.48 mV
    Ca    mean |corr(density,Vmem)| 0.496   mean Vmem sd  0.58 mV

The gradients are equally clean in all three cases, so this is physics rather than a generation
defect: Vmem is governed by `Dm_Na`, so only Nav perturbations move it appreciably. About a third of
spatial perturbations are a genuine generalization test and the rest sit within noise of an
unperturbed tissue. Aggregate test_ood error will be dominated by the easy Cl and Ca cases and
will understate difficulty. Recorded in the dataset card with the instruction that per-type OOD
error must be broken down by perturbed channel.

### Cost

Roughly 30 hours wall clock across two sessions at 12 workers, ~117 s per simulation, on a laptop.
Two interruptions, both recoverable and both recovered without losing a record: a deliberate pause
to apply the OOD fix, and Windows ephemeral port exhaustion caused by the run's own loky worker IPC.

### Coherence State
Tests passing: 92/92. Milestone 2 complete: 12,000-sample dataset with full splits and manifest.
Current tier: 3 | Scaffold: full
Regressions across the entire project: none.

### What remains

Milestones 6 and 7 are untouched: the speed benchmark (PRD sec 7.3), the ablations (sec 7.4), and
validation against published experimental Vmem (sec 4.2, sec 7). And the central one: the model has
never been trained on real BETSE data. Every test to date runs against synthetic fixtures with a
deterministic analytic target. The dataset that makes the actual experiment possible now exists;
the experiment has not been run.

---

## [Session 4] Milestone 5: the model meets real data

### Infrastructure

Session opened with `Host key verification failed`. Cause was my own: I addressed the rig as
`windows-rig@abhiram-lenovo`, which is user `windows-rig` at host `abhiram-lenovo` and bypasses the
`Host windows-rig` block in `~/.ssh/config` entirely. The working form is `ssh windows-rig`
(HostName 100.66.69.108, User abhis). No machine fault; uptime unbroken since 29 July.

### Task: GPU support

`TrainingConfig` gained `device: str = "cpu"` and `Trainer` moves model and batches to it. The
field was appended last so every existing positional and keyword construction in the test suite
still works, and the default preserves prior behaviour exactly.

- Instruction sent to 7B: add one field; then modify `Trainer.__init__`, `_predict`, and both loss
  lines.
- 7B response quality: needed one correction. It kept `params = inspect.signature(...)` but dropped
  the `self.needs_graph = ...` line that consumed it. Fixed by specifying the complete 7-line
  `__init__` body rather than describing the change.
- Tests: 92/92 in 498.6 s, including all five BETSE integration tests.

### Task: scripts/train.py

- 7B response quality: one correction. It wrote `inspect.signature(...)` without `import inspect`.
  I had embedded the import inside a prose code block; it dropped that one line. Failure mode F2.
- Verified by running, not by reading: `NameError` at the real traceback, corrected, re-run.

### Finding: the baseline was too good, and that was the signal

First run. MLP, 2 epochs, 16.5 s, no graph access; returned `test_id` MAE 0.970 mV, R^2 0.980.
A 27K-parameter model with no access to gap junctions clearing the accuracy criterion by 8x after
two epochs is not a success, it is a symptom.

I asked the dataset one question: how much of the target variance is *within* a graph versus
*between* graphs.

| split | per-cell density sd within a tissue | within-tissue Vmem sd | across-tissue Vmem sd |
|---|---|---|---|
| train | **0.000** | 1.33 mV | 14.83 mV |
| test_id | **0.000** | 1.34 mV | 13.92 mV |
| test_ood | 1.523 | 4.22 mV | 16.46 mV |

Every cell in every training tissue carries an identical channel-density vector. `ConfigSampler`
draws one vector per config and tiles it (`np.tile(per_channel, (n_cells, 1))`); only perturbations
introduce per-cell variation, and train/val/test_id are entirely baseline.

The consequence is structural, not statistical. For a spatially uniform tissue every cell reaches
the same voltage, so every `(V_i ??? V_j)` is zero, so the junctional term `sum_j g_ij (V_i ??? V_j)`
vanishes in the bulk for any conductance whatsoever. The gap junctions are physically present
and doing nothing. The in-distribution task is a pointwise regression with ~1.33 mV of
graph-dependent signal available in total, all of it at the tissue boundary.

This is the most important thing found in the project so far and it cost one probe script.
Recommendation recorded in the report: the within-graph vs across-graph target variance ratio
should be a standard check for any graph-learning benchmark.

### Task: scripts/evaluate.py

Degree-stratified error, paired within graph, plus OOD cross-tabulated by family and perturbed
channel.

- 7B response quality: two corrections.
  - It assumed `perturbation_type` was an attribute of the `Data` object. It is not --
    `dataset.py` stores only `config_id` and `is_perturbation`. My spec error, not the 7B's.
  - Told to read files via `sorted(glob.glob(...))`; the exact line was in its prompt; it wrote
    `f"{i}.npz"` instead (F7, invented convention). In the same round it hoisted the paired-analysis
    block out of the split loop and rewired it to iterate `report["splits"].items()`, which holds
    metric dicts rather than prediction arrays (F8, block relocation).
  - Both were fixed in one round by stating the entire `main()` verbatim. The F1 mitigation
    generalizes: never describe where a line goes, state the enclosing function.

### Finding: the degree control came out negative

I predicted the MPNN's advantage would concentrate at boundary cells, where coupling is asymmetric.
Interior cells turned out roughly 3x *harder* than boundary cells; for both models.

| model | split | boundary MAE | interior MAE | paired difference |
|---|---|---|---|---|
| MLP | test_id | 0.324 | 0.949 | 0.625 |
| MPNN | test_id | 0.276 | 0.881 | 0.605 |

The MLP cannot see node degree. Within a uniform tissue it receives identical inputs for every
cell and is mathematically constrained to emit one value. It cannot be exploiting boundary
structure, yet it shows the same asymmetry at the same magnitude. The effect is therefore a
property of the data, not of message passing, and the control fails to support the graph
hypothesis. Mechanism not established; mesh geometry is the leading candidate.

Residual signal worth keeping: the MPNN's *relative* advantage is roughly twice as large at
boundary cells (12-14%) as at interior cells (5.6%).

### Milestone 5 COMPLETE: and the headline is a null

Three seeds, both architectures, identical training path:

| split | MLP | MPNN | difference |
|---|---|---|---|
| test_id | 0.7974 +/- 0.0161 | 0.7669 +/- 0.0116 | ???3.8% |
| test_ood | 1.1900 +/- 0.0529 | 1.1615 +/- 0.0567 | ???2.4% |

Paired per-seed differences (MPNN ??? MLP) are the informative view:

- `test_id`: ???0.0505, +0.0000, ???0.0410 -> mean ???0.0305, sd 0.0264. Two wins, one tie, no losses.
- `test_ood`: ???0.0523, +0.0878, ???0.1209 -> mean ???0.0285, sd 0.1069. The sign flips.

On out-of-distribution data a 664K-parameter graph network is statistically indistinguishable from
a 27K-parameter model that has never seen a gap junction.

How the number moved as the experiment got more honest: 21% -> 6.3% -> 3.8% / null. The 21% was
an unconverged 2-epoch MLP. The 6.3% was a single seed. Both are retained in the report, because
both are the versions that flatter the hypothesis and both were nearly reported.

Claim C2 is not supported. It is also not refuted; sec 11 of the report argues the experiment
as designed could not have tested it. Note that the suite's own
`test_gnn_beats_mlp_on_coupled_data` passes: its fixture draws channels *per cell* and defines the
target as an explicit mixture of a cell's own value and its neighbours' average
(`tests/test_numerical.py:240`, `:262`). The test fixture has the spatial heterogeneity the real
dataset lacks.

---

## [Session 4] Milestone 6: evaluation pipeline

### Housekeeping first

Found a `TestConfigSampler` class written into `nexus/data/config_sampler.py`; test code in a
library module (F6). Functionally inert, since pytest's `testpaths` is `tests/`, and therefore
invisible to the test suite. Caught by reading the source while writing the report. The 7B removed
it and returned the remaining 62 lines byte-identical.

### Task: physics-loss plumbing

`physics_loss_weight: float = 0.0` appended last to `TrainingConfig`; `Trainer` adds
`w * physics_auxiliary_loss(...)` to the training loss only when `w > 0` and the model takes
graph inputs. Validation loss stays pure MAE so that early stopping and checkpoint selection
compare like with like across ablations.

- 7B response quality: first-attempt pass on both files.
- Tests: 87 passed, 5 BETSE deselected.

### Task: ablation infrastructure

`scripts/train.py` extended with `--n-layers`, `--train-size`, `--no-normalize`,
`--physics-weight` and `--tag`. De-normalization is applied through PyG's `transform=` hook, which
`BioelectricDataset.__init__` already accepts.

- 7B response quality: first-attempt pass. Verified by running all four new flags simultaneously:
  `--n-layers 2` gave 234,497 parameters against 663,553 for six layers.

`scripts/run_ablations.py`; first-attempt pass. Eleven configurations, subprocess-driven, skips
runs whose `summary.json` already exists.

### Task: scripts/speed_benchmark.py

- 7B response quality: two corrections, both scoping errors.
  - `UnboundLocalError: cannot access local variable 'stats'`; it used `stats` as a loop variable
    in the summary block, which makes the name local to the entire function and breaks the three
    earlier calls to the module-level `stats()` (F9, variable shadowing).
  - Renaming to `s` but iterating `for s in report["model_seconds"]:` binds `s` to each *key*, a
    string. It had dropped the `s = report["model_seconds"][dev]` line (F2 again). Fixed by
    respecifying as a single-line `for dev, s in ....items()`; removing the two-line idiom
    removed the failure mode.
  - Third round produced correct code but ignored my verbatim print formatting and substituted its
    own (F10). Functionally right, so I did not spend a fourth round on cosmetics.

### Finding: the published BETSE cost was inflated ~2.5x

The 117.2 s/simulation figure quoted everywhere so far was a mean over 13,800 runs under 12-way
parallel load. Measured serially and unloaded, the same simulator takes ~40-50 s for tissues
of 40-232 cells. Throughput and latency are different quantities and I had been reporting one as
the other. This is why the benchmark is being redone to the PRD sec 7.3 protocol: 100 configurations,
run one at a time, median and IQR, CPU and GPU inference reported separately.

Smoke test on two configurations: CPU inference median 11.7 ms (3,359x), CUDA median 4.8 ms
(8,123x), timed region including graph construction and host-to-device transfer.

### Task: scripts/make_figures.py

First-attempt pass. Produced nine PNGs: the in-distribution scatter, the 2x2 perturbation panel,
six spatial error maps (two baseline, four perturbation) and the learning curves. Figures 4 (speed)
and 5 (ablation table) await the running jobs; figure 7 is Milestone 7 data.

### Note on GPU contention

The RTX 4050 has 6.14 GB and the MPNN occupies 5.77 GB while training. Ollama cannot load a 7B
model alongside it, so the executor is unavailable for the entire duration of any training run.
A cleanup request issued during training timed out after 10 minutes with no response. On a
single-GPU rig, code generation and model training are strictly serial and the director must
schedule around it.

The speed benchmark and the ablation sweep are likewise being run serially rather than
concurrently, even though one is CPU-bound and the other GPU-bound. Running them together would
contaminate the per-simulation latency measurement in exactly the way that made the original
117.2 s figure wrong.

### Coherence State
Tests passing: 92/92 (87 + 5 BETSE). Milestone 5 complete. Milestone 6 in progress.
Current tier: 3 | Scaffold: full
Regressions across the entire project: none.
First-attempt passes this session: 6 of 11 tasks. Correction rounds concentrated entirely in the
two scripts with no test coverage (`train.py` 1, `evaluate.py` 2, `speed_benchmark.py` 2);
every test-covered file passed first attempt or needed one round.

### Protocol deviation, logged honestly
Property 5 (scaffold degradation) says three consecutive first-attempt passes earn a trial at
partial scaffold. Four consecutive first-attempt passes occurred this session
(`config.py`, `trainer.py`, `train.py` extension, `run_ablations.py`) and I did not run the
partial-scaffold trial. Delivery pressure displaced the experiment. The scaffold-degradation
dataset is therefore still *n* = 1, which is stated as a threat to validity in the report and
should be corrected in a later session.

### Task: determinism study (unplanned, and the milestone's main finding)

The ablation grid re-ran the base configuration under two extra tags. Comparing them against
`mpnn_seed42` showed three runs of an identical config and seed disagreeing by 0.031 mV; about the
size of the whole MPNN-vs-MLP effect. That was worth measuring rather than noting, so I ran six
replicates plus a controlled mechanism experiment.

Noise floor, MPNN / K=6 / 8000 graphs / seed 42, n=6:
`test_id` 0.7800 +/- 0.0128, `test_ood` 1.1743 +/- 0.0321, early stopping between epoch 48 and 91.

MLP under identical treatment: bit-identical. 0.811936 / 1.216481, epoch 56, twice.

Mechanism, 500 graphs / 10 epochs / two runs per condition:

| condition | run 1 | run 2 | reproducible |
|---|---|---|---|
| MPNN CUDA | 1.106219 | 1.167929 | no |
| MPNN CPU | 1.117110 | 1.056789 | no** |
| MLP CUDA | 1.204118 | 1.204118 | yes |

My first hypothesis was CUDA atomics. The CPU row falsifies it; the MPNN is nondeterministic
on both backends. The MLP result rules out the shared confounders (dataloader shuffle, init,
seeding). Cause is `index_add` in `MPNN.forward`: reduction order is unfixed under parallel
execution on either backend, and floating-point addition is not associative.

Correction I had to make to my own writeup. My first pass compared the *range* of three
replicates (0.0308) against the mean MPNN???MLP difference (0.0305) and concluded the effect was
entirely inside the noise. That compares a range to a mean difference and overstates the noise. With
n=6 the per-run sd is 0.0128, and the two splits separate:

- `test_id`: effect 0.0319 = 2.5x the per-run sd, 6.1x the SE of a 6-run mean. Small but real.
- `test_ood`: effect 0.0422 = 1.3x the per-run sd, and the sign flips across seeds. Not established.

So C2 is weakly supported in distribution and unsupported out of it, not the flat null I stated
first. I had corrected an overstatement in one direction by overstating in the other.

Running tally of how this number has moved: 21% (unconverged baseline) -> 6.3% (single seed) ->
3.8% (three seeds) -> "below the noise floor" (wrong, range-vs-mean) -> 3.9%, 2.5x the per-run sd,
in-distribution only. Five readings, four corrections, and every correction until the last one
moved in the same direction. The pattern is the methodological finding, not any single number.

### Coherence State
Tests passing: 92/92. Milestone 5 complete. Milestone 6 complete at 6 of 7 figures (figure 7 is
blocked on Milestone 7 data, logged as D23/D26).
Current tier: 3 | Scaffold: full
Regressions across the entire project: none.
New project standard: >=3 replicates per reported configuration, since seeds are not the dominant
source of variance (D25).

---

## [Session 4] Milestone 7: experimental validation

### Task: literature curation

Ran five parallel literature searches with an independent adversarial verifier per extracted data
point, each verifier instructed to check that the paper exists, that it reports the value for that
condition, and that the value is a calibrated millivolt measurement rather than uncalibrated dye
intensity. 30 candidates extracted, 29 confirmed, 1 unverifiable, 0 fabricated.

The zero-fabrication result is worth recording: the verification pass was built expecting
plausible-looking non-existent citations, which is the characteristic failure of literature search.
It did not occur. What the verifiers did catch was a subtler class; values quoted from a paper's
*model output* rather than its measurements. Pai et al. 2018 reports simulated and measured
voltages in adjacent sentences, so `source_is_model` became a mandatory schema field.

### Decision: differential validation, not absolute

Predicting absolute Vmem from a prose tissue description requires inventing a channel-density
vector, and the invented vector determines the answer. That is not a test, so I did not do it.

The protocol is a matched-baseline ensemble: select every training tissue whose BETSE
ground-truth mean Vmem lies within +/-3 mV of the measured control, apply the perturbation, and
compare the *distribution* of predicted shifts against the measured shift. Matching on ground truth
rather than model output keeps baseline selection model-independent. The ensemble spread is an
honest expression of the inverse problem's degeneracy; many channel vectors give the same resting
potential and need not respond alike.

### Result: the model predicts ZERO for every gap-junction experiment

| record | dV measured | dV predicted | error |
|---|---|---|---|
| kcnh6 morpholino (zero K_leak) | +20.00 | **+22.23** | 2.23 |
| Ba^2+ frog kidney (zero Kir) | +13.00 | +4.53 | 8.47 |
| Ba^2+ locust tubule (zero Kir) | ???18.00 | +16.65 | 34.65 |
| carbenoxolone 100 uM | +3.10 | **+0.000** | 3.10 |
| carbenoxolone 200 uM | +7.50 | **???0.012** | 7.51 |
| complete uncoupling | +18.80 | **+0.017** | 18.78 |

Stratified against the 5.78 mV threshold (10% of the 57.8 mV experimental range):

- channel blockade, representable channel: 5.35 mV, MEETS (n=2)
- gap-junction blockade: 9.80 mV, fails
- knowingly unrepresentable tissue: 34.65 mV, fails
- all: 12.46 mV, fails

Complete uncoupling (a measured 18.8 mV depolarization) gives a predicted 0.017 mV. That is
not a poor prediction, it is a categorical one: the model has learned that gap junctions do not
affect Vmem.

This was predicted in advance by the sec 11 analysis and is confirmed here from an entirely
independent direction. sec 11 was a statement about within-graph versus across-graph variance in our
own dataset. This is the same statement arriving as a failure against published measurements the
model never saw. A dataset diagnostic and an experimental failure agreeing is much stronger than
either alone, and it makes Experiment C (regenerate training data with spatial structure) the
unambiguous next step rather than one option among several.

### The pre-registered failure fired

`barium_locust_malpighian` was curated deliberately as a case the model should fail: same reagent
and nominal target as the frog kidney record, opposite measured sign, because insect Malpighian
tubules run on an apical V-ATPase we hold at zero. The failure and its reason were written into the
record's `mapping_assumption` field before the model ran. It failed as predicted: 34.65 mV,
wrong sign. Including a case you expect to fail, and saying so first, is cheap insurance against
a mapping protocol that can absorb any result.

### The D1 decision has a cost nobody priced

Three of the PRD's four named validation sources perturb targets we cannot represent; two of them
the channels held at zero under D1. That decision was taken as an accounting convenience to keep an
8-dimensional interface. It disconnects the model from most of the literature that would validate
it. Logged as D27. The general lesson: choose the parameterization backwards from the validation
experiments, not forwards from the simulator's configuration surface.

### Coherence State
Milestone 7 complete. All seven deliverable figures now exist.
Current tier: 3 | Scaffold: full
7B performance this milestone: `nexus/data/experimental.py` and `scripts/validate_experimental.py`
both first-attempt passes; the two longest specifications written this session, both stated as
complete verbatim function bodies. That technique has now produced first-attempt passes on five
consecutive files.


## [2026-08-31] Session 5: Spatial Heterogeneity and v2 Data Generation

### Context
Windows rig went offline for ~2 days (Tailscale disconnect due to idle). Reconnected 2026-08-31.
The sampler heterogeneity change was designed and prompted in Session 4 but never deployed (rig offline).

### Task: Deploy ConfigSampler spatial heterogeneity
- Instruction sent to 7B: complete rewrite of config_sampler.py adding `spatial_heterogeneity=True` kwarg
  to __init__, plus sinusoidal per-cell modulation block in sample() (random wavenumber 1-3, random phase,
  15-50% amplitude per channel, first 6 channels only)
- 7B response quality: first attempt garbled imports and constants (invented 11-element CHANNEL_MAXES,
  wrong indices, wrong imports). Second attempt with fully spelled-out constants: perfect.
- Tests: test_data_pipeline.py 30/30 passed, full suite 87/87 passed (5 betse-deselected)
- Verification: Nav ch per-cell sd = 10.53 with het, 0.0 without. Unmapped ch6 (HKATP) stays 0.0.
  Heterogeneity block runs BEFORE unmapped zeroing and BEFORE perturbation application.
- Notes: the 7B cannot reliably preserve invariant parts of a file while changing specified parts.
  Giving it the ENTIRE file verbatim (constants included) works; telling it to "keep X unchanged" does not.
  This is a fundamental limitation at 7B scale; the attention budget can't maintain both the invariant
  constraint and the modification instruction simultaneously.

### Task: Launch v2 data generation
- Command: `python scripts/generate_dataset.py --out data/synthetic_v2/_staged --n-baseline 11500 --n-per-perturbation 575 --jobs 12 --seed 1000`
- Launched as detached process via Win32_Process.Create (PID 21928)
- Expected duration: ~13h (13800 configs at ~42s/sim serial, 12 workers)
- Sleep disabled on AC power (`powercfg -change -standby-timeout-ac 0`)
- Ephemeral port range already widened from prior session (20000-65535)
- v1 data preserved intact at data/synthetic/ as uniform control arm

### Milestone Status: v2 generation in progress


## [2026-09-02] v2 Training: MPNN vs MLP Head-to-Head

### Task: Retrain both models on v2 heterogeneous dataset
- v2 dataset: 13800 configs, spatial heterogeneity (sinusoidal per-cell modulation)
- Splits: train 8000, val 1000, test_id 1000, test_ood 2000

### MPNN (663K params, 67 epochs, 860s on RTX 4050)
- train MAE 1.049 mV, R2 0.985
- val MAE 1.094 mV, R2 0.981
- test_id MAE 1.070 mV, R2 0.983
- test_ood MAE 1.816 mV, R2 0.961

### MLP (26K params, 35 epochs, 47s on RTX 4050)
- train MAE 2.991 mV, R2 0.919
- val MAE 3.017 mV, R2 0.913
- test_id MAE 3.008 mV, R2 0.915
- test_ood MAE 3.478 mV, R2 0.917

### Result: GRAPH HYPOTHESIS CONFIRMED
- MPNN beats MLP 2.8x on test_id MAE (1.070 vs 3.008 mV)
- MPNN beats MLP 1.9x on test_ood MAE (1.816 vs 3.478 mV)
- With spatial heterogeneity, gap junction topology carries signal the MLP cannot access
- v1 showed no MPNN advantage because uniform tissues gave the graph nothing to learn

### Milestone Status: COMPLETE

## [2026-09-03] v2 Ablation Study Complete ? All 11 Variants

### Results Summary

| tag | arch | K | n_train | norm | test_id MAE | test_ood MAE | epochs |
|---|---|---|---|---|---|---|---|
| mlp_baseline | mlp | - | 8000 | yes | 3.008 | 3.478 | 35 |
| depth_k2 | mpnn | 2 | 8000 | yes | 1.417 | 2.254 | 200 |
| depth_k4 | mpnn | 4 | 8000 | yes | 1.071 | 1.978 | 196 |
| depth_k6 | mpnn | 6 | 8000 | yes | 0.934 | 1.868 | 200 |
| depth_k8 | mpnn | 8 | 8000 | yes | 0.824 | 1.674 | 200 |
| size_1000 | mpnn | 6 | 1000 | yes | 1.273 | 2.327 | 187 |
| size_2000 | mpnn | 6 | 2000 | yes | 1.179 | 2.134 | 131 |
| size_4000 | mpnn | 6 | 4000 | yes | 1.030 | 1.884 | 140 |
| size_8000 | mpnn | 6 | 8000 | yes | 0.945 | 1.713 | 142 |
| no_normalize | mpnn | 6 | 8000 | no | 0.920 | 1.671 | 162 |
| physics_loss | mpnn | 6 | 8000 | yes | 0.924 | 1.810 | 200 |

### Key Findings
- Depth is monotonic on v2 data: more layers = better, coupling length ~4-6 hops
- Data efficiency: 1K samples already beats 8K MLP by 2.4x
- Unnormalized marginally better (0.920 vs 0.945)
- Physics loss: mixed (better test_id, worse test_ood)
- Graph hypothesis confirmed across all ablation variants

### Phase 1 Status: COMPLETE
All planned experiments done. Research report updated. Results synced to Mac.
