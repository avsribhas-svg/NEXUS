# NEXUS Phase 1 — Running Technical Report

**Predicting bioelectric steady states from ion-channel specification, and the
director–executor architecture used to build the system.**

> Live document. Updated as experiments complete. Written to be convertible into a paper:
> Part I is the science, Part II is the methodology experiment that produced the code.
> Section 24 is the changelog. Every number in this document is measured, not estimated;
> where something is an estimate it says so.

Last updated: 2026-08-30, after the first training runs on real BETSE data.

---

# Part I — The NEXUS model

## 1. Problem statement

### 1.1 The long-range question

NEXUS asks whether the computational behavior of a nervous system can be predicted from its
molecular specification. Phase 1 deliberately does not touch neurons. It targets the simpler,
evolutionarily older bioelectric layer: sheets of non-excitable cells whose resting membrane
potentials are set by ion-channel expression and which are electrically coupled to their
neighbors through gap junctions. This layer carries positional and morphogenetic information
— the Levin group's work on *Xenopus* craniofacial patterning and planarian head–tail identity
is the standard reference — and, critically for supervised learning, a mature physics
simulator for it exists.

Phase 1 is the smallest experiment that could falsify the program's core premise. If a
learned model cannot predict a *bioelectric* steady state from a molecular specification, in
a system with clean ground truth and no spiking dynamics, then the far harder neural version
of the question is not worth attempting yet.

### 1.2 The falsifiable claim

> Given per-cell ion-channel densities and the gap-junction adjacency of a tissue, a learned
> model can predict each cell's membrane potential to within 10% of the physiological range,
> generalize to perturbation families it was not trained on, and do so several orders of
> magnitude faster than the simulator.

Four sub-claims, each independently testable:

| # | Claim | Metric | Status |
|---|---|---|---|
| C1 | Accuracy in-distribution | MAE < 10% of Vmem range | **Met, but see §11** |
| C2 | Graph structure is necessary | MPNN beats density-only MLP | **Weakly supported in-distribution (0.03 mV, 2.5× noise); not supported OOD (§10.7)** |
| C3 | Generalization to unseen perturbations | OOD MAE within tolerance, per family | **Met on tolerance, but see §10.5** |
| C4 | Speedup over simulator | inference vs measured BETSE latency | **Met** (§12) |

C1 and C4 are met comfortably. **C2 is not supported**, and §11 argues that the experiment as
designed could not have supported it: the training distribution contains no intra-tissue spatial
variation, so the graph carries almost no information in-distribution. That argument, and the
negative degree-control in §10.4, are the main scientific content of this report so far.

### 1.3 Why a graph network is the right hypothesis class

Vmem in a coupled sheet is not a pointwise function of a cell's own channels. Current flows
between cells through gap junctions, so each cell's potential is pulled toward its neighbors'.
At electrical equilibrium the junctional current into cell *i* balances its transmembrane
current:

```
    Σ_j  g_ij (V_i − V_j)  =  I_membrane(V_i ; channels_i)
```

The left-hand side is a weighted graph Laplacian applied to the voltage field. A
message-passing network whose aggregation runs over the same adjacency, with edge features
carrying `g_ij`, can in principle represent the fixed point of this system: *K* rounds of
message passing approximate *K* steps of a relaxation solver on the Laplacian.

The null hypothesis is that channel densities alone suffice — that the coupling term is
negligible and Vmem is essentially a pointwise function of local channel composition. The
density-only MLP baseline (§7.2) is the direct test of that null.

**Result preview: the null has not been rejected in-distribution, for reasons that turn out
to be a property of how the dataset was sampled rather than a property of the physics.**

---

## 2. Ground truth: the BETSE simulator

### 2.1 What it is

BETSE 1.5.0 (BioElectric Tissue Simulation Engine), Python ≥ 3.11, actively maintained as of
April 2025. It constructs a Voronoi cell cluster from a seeded point lattice and integrates
ion concentrations and membrane voltages using a discrete-exterior-calculus formulation of
electrodiffusion — Nernst–Planck flux for each ion species coupled to a Poisson solve for the
electric field, with explicit representations of voltage-gated and leak channel conductances,
ATP-driven pumps (Na⁺/K⁺-ATPase, H⁺/K⁺-ATPase, V-ATPase), and gap-junction coupling with
voltage-dependent gating.

It is CLI- and YAML-driven, in four stages:

| Stage | What it does | What we do with it |
|---|---|---|
| `betse config` | scaffolds a default YAML | starting point we then patch |
| `betse seed` | builds the Voronoi mesh | **we read the mesh here to learn cell indices** |
| `betse init` | initializes concentrations to equilibrium | runs with our patched profiles |
| `betse sim` | integrates the dynamics | produces the Vmem field we label with |

### 2.2 Integration approach

We drive BETSE as a subprocess per simulation rather than through its Python API. The reasons
are pragmatic and worth recording: BETSE's internal state is global and not reentrant, its
API surface is undocumented for programmatic use, and subprocess isolation means a simulation
that diverges or segfaults cannot corrupt the generator. The cost is process startup overhead
(~4 s of the 117 s mean) and the need to marshal results through pickle files.

Per simulation, `BETSEGenerator.run(config, timeout_s, capture_timeseries)` does:

1. `tempfile.mkdtemp` a scratch working directory.
2. `betse config sim\config.yaml` to scaffold. **The separator must be a backslash**; BETSE on
   Windows rejects `sim/config.yaml`. This cost a full debugging cycle.
3. Load the YAML with `ruamel.yaml` (round-trip loader, to preserve BETSE's comments and
   anchors), patch it, write it back.
4. `betse seed config.yaml` — after which `INITS/world_*.betse.gz` contains the mesh.
5. Read `cells.cell_centres`, resample the requested per-cell densities onto the actual mesh
   (§5.3), bin cells into density groups, and **append** indices-based tissue profiles.
6. `betse init`, then `betse sim`.
7. Unpickle `SIMS/*.betse.gz`, extract `sim.vm_ave * 1000` (V→mV),
   `cells.cell_centres * 1e6` (m→µm), `cells.cell_nn_i` masked for self-loops, and
   `sim.gjopen`.
8. Emit one `.npz` record; delete the scratch directory.

The whole run is wrapped in a `try/except` returning `None` on any failure, with a decrementing
timeout budget carried across the three stages so a single hung stage cannot exceed the total.

### 2.3 What BETSE exposes, and what it does not

The specification fixes an 8-dimensional per-cell channel vector:

```
[Nav, Kir, K_leak, Ca, Cl, NaKATP, HKATP, VATP]
```

BETSE has no equivalent for **HKATP** (H⁺/K⁺-ATPase) or **VATP** (V-ATPase) in the
configuration surface we drive. Three options were available: drop to a 6-dimensional vector,
find an indirect proxy, or hold the two columns at zero.

**We hold columns 6 and 7 at exactly zero.** The reasoning:

1. The acceptance test suite hardcodes `N_CHANNELS = 8` in five separate files. Changing the
   dimension would mean editing the spec, which is the one artifact treated as immutable.
2. Phase 2 intends to feed a transcriptomic encoder into the same input slot. An encoder
   trained against a 6-dimensional interface would have to be re-fit for an 8-dimensional one.
3. The implementation cost is a single conditional in the sampler
   (`zero_unmapped_channels=True`, `UNMAPPED_CHANNEL_INDICES = (6, 7)`).

**Consequence that must be reported:** two of eight input features are identically zero across
all 12,000 records. They carry no signal. Any parameter-efficiency or feature-importance claim
must state that the effective input dimension is six.

---

## 3. Deriving the physics mapping by measurement

### 3.1 Why measurement rather than reading the source

The correspondence between our abstract "channel density" units and BETSE's configuration
parameters is not documented in a form that could be trusted. BETSE's YAML exposes both
*membrane diffusion constants* (`Dm_Na`, `Dm_K`, `Dm_Cl`, `Dm_Ca` — background permeabilities)
and *named channel objects* (`Kir2p1`, `Nav1p3`, `K_Leak`, each with a `max Dm`). Which knob
actually controls resting potential was an empirical question.

We answered it by sweeping single parameters over their plausible ranges and recording the
resulting Vmem field.

### 3.2 Sweep results

| Swept parameter | Range | Vmem response | Interpretation |
|---|---|---|---|
| `Dm_Na` | 1e-18 → 4e-18 | **57 mV swing** | Dominant control on resting Vmem |
| `Dm_K` | swept jointly | **1–3 mV** | Effectively inert in this configuration |
| Kir2p1 `max Dm` | 0 → 1e-15 | **−44.6 → −85.2 mV** | Strong hyperpolarizing control |
| Nav1p3 channel | on/off | **no effect** | Correct physics — see below |

Two of these deserve comment.

**Nav1p3 is inert at rest, and that is correct.** A voltage-gated sodium channel is closed at
a resting potential of −63 mV; it activates near −40 mV. Its contribution to a *resting*
potential is genuinely zero. This looked like a bug for some hours. It is not. The consequence
is that our `Nav` feature must act through the *background* sodium permeability `Dm_Na`, not
through BETSE's gated Nav channel object. The feature name is therefore slightly misleading:
it represents total resting sodium permeability, not Nav channel density.

**`Dm_K` is inert because potassium is controlled elsewhere.** Potassium permeability in this
configuration is dominated by the `Kir2p1` and `K_Leak` channel objects, whose `max Dm` values
are 3e-16 and 2e-17 — 300× and 20× the background `DM_BASE` of 1e-18. Modulating the
background K permeability against those is invisible. An early version of the mapping routed
`Kir` and `K_leak` through `Dm_K`; measurement showed it did nothing, and the mapping was
rewritten to drive the channel objects directly.

### 3.3 The final mapping

`nexus/data/betse_config.py`, `channel_densities_to_betse(densities) -> dict`. Let
`frac(c) = clip(density_c / CHANNEL_MAXES[c], 0, FRAC_MAX)` with `FRAC_MAX = 4.0`:

| BETSE parameter | Formula | Range over frac ∈ [0,1] |
|---|---|---|
| `Dm_Na` | `DM_BASE · (1 + 3·frac(Nav))` | 1e-18 → 4e-18 |
| `Dm_K` | `DM_BASE` (constant) | — |
| `Dm_Cl` | `DM_BASE · (0.1 + 1.9·frac(Cl))` | 1e-19 → 2e-18 |
| `Dm_Ca` | `DM_BASE · (0.1 + 1.9·frac(Ca))` | 1e-19 → 2e-18 |
| `alpha_NaK` | `ALPHA_NAK_BASE · (0.5 + 1.0·frac(NaKATP))` | 5e-8 → 1.5e-7 |
| Kir2p1 `max Dm` | `KIR_MAX_DM · frac(Kir)` | 0 → 3e-16 |
| K_Leak `max Dm` | `KLEAK_MAX_DM · frac(K_leak)` | 0 → 2e-17 |

Gap junctions map through surface area rather than a conductance parameter:

```
gj_surface_area = GJ_SURFACE_CLOSED + (gj_conductance/50) · (GJ_SURFACE_OPEN − GJ_SURFACE_CLOSED)
                = 1e-9 + frac · (1e-7 − 1e-9)
```

World size uses hexagonal-packing scaling: `world_size = 2.28 · r · √n`, with `WORLD_ALPHA =
2.28` chosen so that a seeded lattice yields approximately the requested cell count. In
practice BETSE returns a somewhat different count than requested — the mesh is generated by
Voronoi tessellation of a disordered lattice and boundary cells are culled — so the pipeline
always uses the **actual** returned cell count, never the requested one. Final counts span
40–490 with median 150 against a requested range of 50–500.

### 3.4 The `FRAC_MAX = 4.0` clamp is load-bearing for OOD

`frac` is clamped at 4.0, not 1.0. This exists because the `exogenous_expression` perturbation
deliberately sets a patch of cells to 4× the nominal channel maximum. Without the clamp those
cells would map to arbitrarily large diffusion constants and destabilize the integrator; with
a clamp at 1.0 the perturbation would be silently erased. Setting it at 4.0 admits the
perturbation while bounding it.

This has a downstream consequence for the learned models that is easy to miss and matters a
great deal (§11.4): the dataset normalizer divides channel densities by `CHANNEL_MAXES`, so
`exogenous_expression` cells arrive at the network with **input features of 4.0 in normalized
units, when every training example lies in [0, 1]**. That perturbation family is therefore a
test of input-range extrapolation as much as of physical generalization.

---

## 4. There is no reachable steady state

### 4.1 The original criterion was unsatisfiable

The specification asked for *steady-state* Vmem with a convergence criterion on
`max |dVmem/dt|`. Direct measurement showed no such state is reachable over this parameter
space in any tractable integration window. Two distinct processes run on separated timescales:

1. **A fast electrical transient**, essentially complete within 4–6 simulated seconds, in
   which membrane capacitance charges to the balance of channel and pump currents. This is the
   process determined by channel expression and gap-junction topology — the quantity the Phase
   1 claim is actually about.

2. **A slow secular concentration drift.** The Na⁺/K⁺-ATPase runs continuously, moving ions
   against their gradients at a rate set by `alpha_NaK`. Intracellular concentrations therefore
   keep changing on a timescale far longer than any simulation we can afford, and Vmem drifts
   with them. There is no fixed point. The system is a driven, dissipative one; a true steady
   state would require the pump flux to balance passive leak exactly, which happens only for a
   measure-zero subset of the parameter space.

### 4.2 What we did about it

**The prediction target was renamed from "steady-state Vmem" to "Vmem after 5 s of
equilibration."** This is honest, it is reproducible (a fixed integration time is a
well-defined operator on the configuration), and it captures the physically meaningful fast
process. It is not a steady state and no claim in any writeup may describe it as one. The
`.npz` field retains the key `vmem_steady_state` for compatibility with the frozen test suite;
this is a naming artifact, documented here so it does not propagate into a paper as a claim.

### 4.3 A measurement error worth recording

An early convergence check reported `max_dvmem ≈ 1e-5` and was quoted twice as evidence of
convergence before it was caught. The measurement had been taken during the wrong phase of the
simulation — after `init` rather than during `sim` — where the field is quiescent by
construction. It does not support a steady-state claim. It is recorded here because the same
error is easy to repeat: BETSE's staged execution makes it possible to measure the right
quantity at the wrong time and get a plausible number.

---

## 5. Dataset design

### 5.1 Parameter space and sampling

`ConfigSampler` draws from an 11-dimensional unit hypercube using Latin Hypercube Sampling
(`scipy.stats.qmc.LatinHypercube(d=11)`), then maps each dimension to its physical range:

| Dim | Parameter | Mapping | Range |
|---|---|---|---|
| 0 | `n_cells` | log-uniform | 50 – 500 |
| 1 | `cell_radius` | uniform | 5 – 15 µm |
| 2 | `gj_conductance` | uniform | 0 – 50 nS |
| 3–10 | 8 channel densities | uniform × `CHANNEL_MAXES` | see below |

`CHANNEL_MAXES = [50, 30, 20, 10, 15, 30, 10, 10]` for
`[Nav, Kir, K_leak, Ca, Cl, NaKATP, HKATP, VATP]`. Columns 6 and 7 are then zeroed.

Cell count is sampled **log-uniformly** rather than uniformly, so that small tissues are not
crowded out — a uniform draw over 50–500 puts 90% of the mass above 100 cells, which would
have left the small-tissue regime (where boundary effects dominate) badly undersampled.

LHS was chosen over uniform random for coverage: with *n* samples it guarantees exactly one
sample per stratum in each marginal dimension, which matters when 12,000 samples must cover an
11-dimensional space. The acceptance suite tests this directly
(`test_lhs_coverage`: ≥ 35 of 50 bins occupied in a 50-sample draw).

### 5.2 The four perturbation families

These define the out-of-distribution split. All are *held out entirely from training*.

| Family | Operation | Intended test |
|---|---|---|
| `channel_blockade` | one of the 6 mapped channels driven to 0 for all cells | pharmacological knockout |
| `gj_blockade` | `gj_conductance = 0` | decoupling — isolates the graph term |
| `spatial_gradient` | linear density ramp 0 → max across the tissue in *x* | morphogen-gradient analogue |
| `exogenous_expression` | 25% of cells set to 4× max density | localized transfection analogue |

Spatial families draw their target channel only from `{Nav, Ca, Cl}`
(`SPATIAL_CHANNEL_INDICES = (0, 3, 4)`) because those are the three that map to per-cell
membrane diffusion constants and can therefore be varied cell-by-cell. `Kir` and `K_leak` act
through named channel objects whose `max Dm` is a per-*profile* property, and `NaKATP` through
`alpha_NaK`, which is a global internal parameter. This is a limitation of the simulator
interface, not a design choice, and it directly produces the OOD heterogeneity in §5.6.

### 5.3 Per-cell spatial structure required a non-obvious mechanism

BETSE assigns diffusion constants per *tissue profile*, not per cell, which initially appeared
to make spatial gradients impossible. The mechanism that works:

> Tissue profiles with `cell targets: {type: indices}`, written into the YAML **between the
> `seed` and `init` stages** — after the mesh exists (so cell indices are known and stable)
> but before the simulation initializes.

The pipeline bins cells into 8 groups by the maximum-variance density column
(`group_cells_by_density`, quantile split via `np.array_split` on the sorted order) and emits
one profile per group with that group's mean densities. Eight groups is a compromise: more
profiles resolve the gradient better but BETSE's per-profile setup cost is superlinear.

Mapping the requested gradient onto the actual mesh is `resample_densities_to_mesh`. The
sampler produces a density array indexed 0…n_requested−1; the mesh returns n_actual cells at
arbitrary positions. The resampler does `np.lexsort((y, x))` — sorting cells primarily by *x*
— and maps rank *r* to source row `⌊r · n_requested / n_actual⌋`. An index-ordered ramp in the
sampler therefore becomes a spatially-ordered ramp in *x* on the mesh.

Two failure modes were hit here:

- **`del prof_list[:]` breaks BETSE.** Replacing the profile list deletes the shipped `Spot`
  profile, which BETSE's `general network` block references by name, producing
  `KeyError: 'Spot'`. Profiles must be **appended**; the config specifies that later profiles
  override earlier ones for overlapping cells.
- **The path separator.** `betse config sim/config.yaml` fails on Windows; only
  `sim\config.yaml` works.

**Production verification.** Measured on generated records, not on a test fixture:
`corr(x, density) = 0.997` spanning the full channel range (Nav 0 → 49.8, Cl 0 → 15.0,
Ca 0 → 9.98). A baseline configuration run through the identical code path was **bit-identical**
to one run before the profile machinery existed (`max |ΔVmem| = 0.000e+00`), confirming the
mechanism is inert when no spatial variation is requested.

### 5.4 Record schema

One `.npz` per tissue:

| Key | Shape | Units |
|---|---|---|
| `config_id` | scalar str | — |
| `n_cells` | scalar int | — |
| `cell_positions` | (n, 2) float32 | µm |
| `channel_densities` | (n, 8) float32 | nominal density units |
| `edge_index` | (2, m) int64 | COO, self-loops removed |
| `conductances` | (m,) float32 | nS |
| `vmem_steady_state` | (n,) float32 | mV *(see §4.2 on the name)* |
| `is_perturbation` | scalar bool | — |
| `perturbation_type` | scalar str or None | — |

Edges come from `cells.cell_nn_i` (BETSE's nearest-neighbor adjacency), masked to drop
self-loops. Conductances are `sim.gjopen · gj_conductance` — that is, the *nominal* conductance
scaled by BETSE's voltage-dependent gate state at the end of the simulation, so an edge whose
junction has closed carries a correspondingly smaller weight.

### 5.5 Generation campaign

| | |
|---|---|
| Simulations attempted | 13,800 |
| Failures | **0** |
| Mean wall clock per simulation, **under 12-way parallel load** | 117.2 s |
| Median wall clock per simulation, **serial and unloaded** | **~42 s** (§12) |
| Parallelism | 12 joblib/loky workers |
| Records finalized | 12,000 |
| Surplus (unused) | 1,800 |
| Total size | 0.11 GB |
| Campaign wall clock | ≈ 39 h |

Parallelism used `joblib` with `return_as="generator_unordered"` so results stream to disk as
they complete rather than accumulating in memory; at 12 workers × up to 500-cell meshes the
batched alternative would have been memory-bound.

**Throughput was measured, not inferred.** Per-simulation latency suggested saturation at 6
workers. The measured curve:

| Workers | Throughput |
|---|---|
| 6 | 267 sims/hr |
| 12 | **352 sims/hr** |
| 18 | 352 sims/hr |

Believing the latency-based inference would have cost roughly 16 additional hours of wall
clock. The lesson generalizes: per-task latency under load is a bad proxy for system
throughput when the tasks are I/O- and startup-bound rather than compute-bound.

**The predicted failure mode did not occur.** The specification anticipated a 5–15% failure
rate concentrated in `gj_blockade`, on the reasoning that zero gap-junction conductance makes
the gap-junction Laplacian singular and BETSE precomputes a dense inverse of it. The observed
failure rate was zero, in 575/575 `gj_blockade` runs. The reason is incidental: our conductance
mapping floors the surface area at `GJ_SURFACE_CLOSED = 1e-9` rather than true zero, so cells
become nearly isolated while the matrix stays invertible. A constant chosen for physical
plausibility silently protected the numerics.

This is worth stating plainly in any writeup because it means **`gj_blockade` is a
near-blockade, not a blockade** — residual conductance is 1% of the open value. The
perturbation is real but weaker than its name implies.

### 5.6 Splits

| Split | N | Composition |
|---|---|---|
| `train` | 8,000 | baseline (unperturbed) |
| `val` | 1,000 | baseline |
| `test_id` | 1,000 | baseline |
| `test_ood` | 2,000 | 500 each of the four perturbation families |

Verified programmatically: no `config_id` appears in two splits; `train` is entirely baseline;
`test_ood` is entirely perturbation; every sampled record carries the eight required keys with
Vmem inside [−120, 60] mV.

### 5.7 OOD difficulty is highly heterogeneous — read before reporting any OOD number

Measured across `spatial_gradient` records, grouped by which channel the perturbation targeted:

| channel | mean \|corr(x, density)\| | mean \|corr(density, Vmem)\| | mean within-tissue Vmem sd |
|---|---|---|---|
| **Nav** | 0.996 | **0.965** | **10.58 mV** |
| Cl | 0.996 | 0.533 | 0.48 mV |
| Ca | 0.997 | 0.496 | 0.58 mV |

The gradients are clean in all three cases — the generation is not at fault. **The disparity
is physics.** Vmem is governed by `Dm_Na` (§3.2); a Cl or Ca gradient shifts Vmem by ~0.5 mV,
which is under 3% of the 10%-of-range accuracy target and within noise of an unperturbed
tissue.

Since spatial families draw uniformly from `{Nav, Ca, Cl}`, **approximately one third of
spatial perturbations constitute a genuine generalization test and two thirds are
indistinguishable from baseline.** An aggregate `test_ood` MAE will be dominated by the easy
cases and will understate difficulty by roughly a factor of three on the spatial families.

**Requirement for reporting:** per-family OOD error must be broken down by perturbed channel.
The channel is recoverable from each record as the column of maximum variance in
`channel_densities`.

---

## 6. Data pipeline

`BioelectricDataset(root, split)` subclasses `torch_geometric.data.InMemoryDataset`. It reads
`root/{split}/*.npz` and produces `Data` objects with:

| Field | Content | Normalization |
|---|---|---|
| `x` | (n, 8) channel densities | ÷ `CHANNEL_MAXES` → [0, 1] in-distribution |
| `edge_index` | (2, m) int64 | — |
| `edge_attr` | (m, 1) conductances | ÷ 50 → [0, 1] |
| `y` | (n,) Vmem | **unnormalized, in mV** |
| `pos` | (n, 2) positions | unnormalized, µm |

Two deliberate choices:

**Targets are left in millivolts.** The loss is therefore directly interpretable as
"millivolts of error," and the 10%-of-range acceptance threshold can be read off without
rescaling. The cost is that the network must learn an output scale; this is handled
architecturally (§7.1) rather than by target normalization.

**Normalization constants are fixed, not fitted.** `CHANNEL_MAXES` and `GJ_MAX` are the
sampler's own bounds, not statistics of the training set. This means the transform is
identical for train and OOD data and there is no train-statistics leakage — but it also means
`exogenous_expression` inputs land at 4.0 rather than being squashed into range (§3.4).

Processing is cached to `root/processed/{split}.pt` on first load.

---

## 7. Model architectures

### 7.1 MPNN — 663,553 parameters

```
node encoder:  8 → 64 → 128         (Linear, LayerNorm, ReLU) ×2
edge encoder:  1 → 32 → 64          (Linear, LayerNorm, ReLU) ×2

× 6 message-passing layers:
    msg   = MLP_k( [ h_dst ‖ h_src ‖ e ] )        320 → 128 → 128
    agg   = index_add(msg, dst) / clamp(deg, min=1)     mean aggregation
    h     = LayerNorm_k( h + MLP'_k( [ h ‖ agg ] ) )    256 → 128 → 128, residual

decoder:       128 → 64 → 1
output:        (decoder(h)).squeeze(-1) · 30.0 − 50.0
```

Design decisions and their reasons:

- **Mean aggregation, not sum.** Cell degree varies with position (boundary cells have fewer
  neighbors) and tissues vary in size from 40 to 490 cells. Sum aggregation would make node
  representations scale with degree, conflating "highly connected" with "large input." Mean
  keeps the message scale degree-invariant; the degree information the model needs is
  recoverable from the boundary geometry itself. The degree is clamped at 1 so isolated nodes
  (possible under `gj_blockade`) do not divide by zero.
- **Residual connections plus LayerNorm at every layer.** Six rounds of message passing is
  deep enough for oversmoothing to be a real risk — repeated neighborhood averaging drives all
  node representations toward the graph mean, which is precisely the failure that would make
  the MPNN collapse to the MLP's behavior. The residual gives every layer an identity path.
- **Edge features are encoded once, outside the loop.** Conductance does not change between
  layers, so re-encoding it per layer would only add parameters.
- **The fixed output affine map `×30 − 50`.** This is an architectural prior: it places the
  network's initialization near −50 mV, inside the physiological range, and scales gradients
  so that a unit change in the decoder output corresponds to 30 mV. Without it the network
  starts near 0 mV — outside the biological range — and spends its early epochs traversing
  the offset. The constants are fixed, not learned, so they inject no fitted information.

**Message-passing depth as a modeling claim.** Six layers means each cell's prediction can
depend on cells up to six hops away. For a Voronoi mesh with ~6 neighbors per cell that is a
receptive field of roughly 100 cells — comparable to a whole small tissue and a meaningful
fraction of a large one. If the true electrical coupling length is longer than six hops, the
architecture is under-powered; this is a testable hypothesis that has not yet been tested (a
depth sweep is Experiment F, §13).

### 7.2 Baseline MLP — 26,625 parameters

```
8 → 128 → 128 → 64 → 1   (Linear, LayerNorm, ReLU), same ×30 − 50 output map
```

Applied per node. **It receives neither `edge_index` nor `edge_attr`.** This is the null model:
it can express only a pointwise function of a cell's own channel densities.

Two consequences that make it a sharp test:

1. Because baseline tissues are spatially uniform (§11.1), every cell in a tissue presents the
   MLP with an identical input vector, so the MLP is *mathematically constrained* to emit an
   identical output for every cell in that tissue. Its best possible prediction is the tissue
   mean, and its irreducible error is exactly the within-tissue standard deviation of Vmem.
2. **The MLP is entirely blind to gap-junction conductance**, which lives only in `edge_attr`.
   Any performance it achieves is achieved without that variable.

### 7.3 Losses

`mae_loss(pred, target) = mean|pred − target|`. L1 rather than L2, chosen because the target
distribution has heavy tails — a handful of extreme configurations (fully blocked Kir with high
Na permeability) produce Vmem near the boundary of the physiological range, and MSE would let
those dominate the gradient.

`physics_auxiliary_loss(pred, edge_index, conductances)` computes, for each node, the net
junctional current implied by the predicted voltage field, and returns the mean squared
residual:

```
    I_ij  = g_ij (V_i − V_j)                 per edge
    net_i = Σ_j I_ij                         index_add over source nodes
    loss  = mean(net²)
```

This is the left-hand side of the equilibrium relation in §1.3. Penalizing it pushes
predictions toward voltage fields that are self-consistent under the graph Laplacian. It
returns exactly `0.0` for empty edge sets.

**It is implemented and tested but has not yet been used in a training run.** The reason is
§11: on spatially uniform tissues the junctional residual is near zero everywhere by
construction, so the term would contribute almost no gradient. It becomes meaningful only once
the training set has spatial structure.

---

## 8. Training protocol

| Setting | Value | Note |
|---|---|---|
| Optimizer | AdamW | |
| Learning rate | 1e-3 | cosine annealed to 1e-5 over `max_epochs` |
| Weight decay | 1e-4 | decoupled |
| Batch size | 32 graphs | PyG batching — one large block-diagonal graph |
| Gradient clipping | norm 1.0 | |
| Max epochs | 200 | |
| Early stopping | patience 20 on validation MAE | best state restored at the end |
| Loss | MAE in mV | |
| Seeds | 42 / 137 / 256 | only 42 complete so far |
| Device | CUDA (RTX 4050 Laptop, 6 GB) | |

The `Trainer` detects whether its model needs graph inputs by inspecting the forward signature
(`len(inspect.signature(model.forward).parameters) >= 3`), so the identical training path
serves both architectures — an important control, since it removes "the baseline was trained
differently" as a confound.

Checkpointing writes `best.pt` whenever validation improves, and the best state dict is
restored into the model before `train()` returns, so the object handed to evaluation is the
best-validation model rather than the last-epoch model.

**Resource note:** the MPNN saturates the 4050 at 82–86% utilization and 5.77 GB of 6.14 GB.
This is at the edge; a larger batch or a deeper network would not fit. It also means the GPU
cannot host anything else during training — including the local code-generation model
(§20.3).

---

## 9. Evaluation protocol

| Metric | Definition |
|---|---|
| MAE | `mean|pred − true|`, in mV |
| R² | `1 − SS_res/SS_tot`; returns 0.0 when `SS_tot = 0` |
| Accuracy threshold | `0.10 · (vmem_max − vmem_min)` — the 10%-of-range criterion |
| Per-group MAE | MAE within each group of a supplied grouping array |

The R² guard matters: evaluated on a *single* uniform tissue, `SS_tot` is near zero and R² is
numerically meaningless. All R² figures in this report are computed over a whole split, where
`SS_tot` is dominated by across-tissue variance.

**Every metric in §10 is computed over concatenated per-cell predictions across a full split**,
so a 490-cell tissue contributes 490 residuals and a 40-cell tissue contributes 40. This
weights large tissues more heavily. A per-tissue-averaged variant would weight tissues equally;
both are defensible and the paper should report which is used. We use per-cell because the
claim is about per-cell prediction.

---

## 10. Results

### 10.1 Acceptance suite

**92 / 92 tests pass in 498.6 s**, including all five BETSE integration tests, which run real
simulations rather than fixtures. Coverage:

| File | Tests | Domain |
|---|---|---|
| `test_data_pipeline.py` | 30 | validation, sampling, dataset loading, normalization, splits |
| `test_model.py` | 24 | shapes, gradient flow, permutation equivariance, edge sensitivity |
| `test_evaluation.py` | 13 | metrics, grouping, figure generation |
| `test_numerical.py` | 7 | physics invariants, symmetry, topology sensitivity |
| `test_training.py` | 10 | loss decrease, overfitting, checkpoint round-trip, early stopping, LR |
| `test_integration.py` | 8 | end-to-end pipeline (3), BETSE integration (5) |

Notable among these: `test_model.py` verifies that permuting node order permutes outputs
identically (equivariance) and that changing `edge_index` changes the MPNN's output (the model
genuinely reads topology); `test_numerical.py` verifies that a symmetric input configuration
yields a symmetric Vmem field. These are the tests that would catch a GNN that had silently
degenerated into a per-node MLP.

### 10.2 Baseline MLP, preliminary 2-epoch run

The first run on real data, kept here because it is what prompted the analysis in §11.
Seed 42, 2 epochs, CPU, 16.5 s, 26,625 parameters, no graph access. **Not converged** — the
converged comparison is §10.3.

| split | MAE (mV) | R² |
|---|---|---|
| train | 0.992 | 0.9795 |
| val | 0.979 | 0.9820 |
| test_id | 0.970 | 0.9802 |
| test_ood | 1.441 | 0.9674 |

The 10%-of-range threshold on this dataset is approximately 8 mV. **The density-only baseline
clears the Phase 1 accuracy criterion by a factor of eight, after two epochs, without ever
seeing the graph.**

### 10.3 MPNN vs baseline — three seeds, both converged

Identical training path, early stopping on validation MAE, RTX 4050. Seeds 42 / 137 / 256:

| arch | seed | epochs | test_id MAE | test_ood MAE |
|---|---|---|---|---|
| MLP | 42 | 56 | 0.8119 | 1.2165 |
| MLP | 137 | 77 | 0.7801 | 1.1290 |
| MLP | 256 | 54 | 0.8001 | 1.2245 |
| MPNN | 42 | 91 | 0.7615 | 1.1642 |
| MPNN | 137 | 66 | 0.7802 | 1.2168 |
| MPNN | 256 | 95 | 0.7591 | 1.1035 |

Aggregated (mean ± sample sd over 3 seeds):

| split | MLP (26,625 params) | MPNN (663,553 params) | difference |
|---|---|---|---|
| `test_id` | 0.7974 ± 0.0161 | **0.7669 ± 0.0116** | **−0.0305 (−3.8%)** |
| `test_ood` | 1.1900 ± 0.0529 | 1.1615 ± 0.0567 | −0.0285 (−2.4%) |

Paired per-seed differences (MPNN − MLP), which control for seed-specific data ordering:

| split | seed 42 | seed 137 | seed 256 | mean | sd of the differences |
|---|---|---|---|---|---|
| `test_id` | −0.0505 | **+0.0000** | −0.0410 | −0.0305 | 0.0264 |
| `test_ood` | −0.0523 | **+0.0878** | −0.1209 | −0.0285 | 0.1069 |

**This is the headline result and it is a null on the out-of-distribution split.**

- **`test_id`:** the MPNN wins on two seeds and exactly ties on the third. It never loses. The
  direction is consistent, and a 3.8% mean advantage is plausibly real — but the effect is
  0.03 mV against a seed-to-seed spread of 0.016 mV, with *n* = 3. It is suggestive, not
  established.
- **`test_ood`:** **the sign flips across seeds** (−0.052, +0.088, −0.121). The mean difference
  is −0.0285 against a paired sd of 0.107 — well inside one standard deviation of zero. There
  is no detectable advantage to having the graph on out-of-distribution data.

A 664K-parameter network with full access to gap-junction topology and conductance is
statistically indistinguishable from a 27K-parameter model that has never seen a gap junction,
on the split that was supposed to demonstrate the value of topology. Claim **C2 is not
supported**, and §11 argues the experiment as designed could not have supported it.

**Two cautions about how easy this was to get wrong.** The seed-42-only comparison gave 6.3% and
4.3%, which reads as a modest but real win. The 2-epoch MLP run before that gave 21%, which
reads as a decisive win. Both are in this report (§10.2 and the changelog) because the
unconverged and single-seed versions are the ones that flatter the hypothesis, and both were
numbers this project came close to reporting.

### 10.4 Error by node degree — a control that came out negative

`scripts/evaluate.py` groups per-cell error by node degree. In a Voronoi mesh, degree 6 is an
interior cell and degree ≤ 5 is a boundary cell. Because boundary cells are exactly where
gap-junction coupling is asymmetric, this was intended as a direct test of whether the MPNN
learned coupling: if its advantage concentrated at the boundary, that would be evidence.

`test_id`, per-cell MAE by degree:

| degree | n cells | MLP MAE | MPNN MAE | MPNN advantage |
|---|---|---|---|---|
| 2 | 673 | 0.337 | 0.290 | 13.9% |
| 3 | 15,286 | 0.293 | 0.256 | 12.6% |
| 4 | 17,570 | 0.300 | 0.264 | 11.9% |
| 5 | 10,632 | 0.326 | 0.288 | 11.7% |
| **6 (interior)** | **143,406** | **0.968** | **0.914** | **5.6%** |

Because degree correlates with tissue size, this is also computed **paired within each graph**,
so between-tissue differences cannot contribute:

| model | split | boundary MAE | interior MAE | mean paired difference |
|---|---|---|---|---|
| MLP | `test_id` | 0.324 | 0.949 | 0.625 |
| MPNN | `test_id` | 0.276 | 0.881 | **0.605** |
| MLP | `test_ood` | 0.973 | 1.376 | 0.403 |
| MPNN | `test_ood` | 0.881 | 1.311 | **0.430** |

**Interior cells are ~3× harder than boundary cells for both models.** This was the opposite of
the prediction, and the control that interprets it is the MLP column: **the MLP cannot see node
degree at all.** Within a uniform tissue it receives identical inputs for every cell and is
mathematically constrained to emit one value. It therefore cannot be exploiting the boundary
structure — yet it shows the same asymmetry at nearly the same magnitude (paired difference
0.625 vs the MPNN's 0.605).

**Conclusion: the boundary/interior asymmetry is a property of the data, not of message passing.**
Since the MLP emits a single value *c* per tissue, its per-cell error decomposes as
|V_cell − c|; the observed pattern means *c* sits close to the boundary cells' potential while
interior cells carry more spread around it. In other words, within a spatially uniform tissue the
**interior** Vmem field is less uniform than the rim — which is where the 1.33 mV of within-tissue
variance in §11.1 actually lives. The mechanism is not established; mesh geometry (cell area and
surface-to-volume ratio varying across the Voronoi tessellation) is the leading candidate and is
testable directly from `cell_positions`.

The one piece of evidence that survives for C2: **the MPNN's relative advantage is roughly twice
as large at boundary cells (12–14%) as at interior cells (5.6%)**, and it narrows the
boundary/interior gap slightly in-distribution. That is the pattern topology-awareness would
produce. But the absolute effect is 0.04 mV, it is a single seed, and it goes the other way on
`test_ood` (0.430 vs 0.403). It is suggestive at best and must not be reported as a positive
result.

### 10.5 OOD error by perturbation family and perturbed channel

Aggregating `test_ood` into a single number hides everything that matters (§5.7). Broken out:

| family / channel | n cells | MLP MAE | MPNN MAE | MPNN advantage |
|---|---|---|---|---|
| `exogenous_expression` / **Nav** | 28,769 | 3.693 | **3.304** | 10.5% |
| `channel_blockade` / K_leak | 15,442 | 1.377 | 1.213 | 11.9% |
| `channel_blockade` / Kir | 13,399 | 1.324 | 1.133 | **14.4%** |
| `exogenous_expression` / Cl | 34,153 | 1.416 | 1.270 | 10.3% |
| `channel_blockade` / Nav | 14,902 | 1.321 | 1.302 | 1.4% |
| `gj_blockade` / — | 95,120 | 1.115 | **1.000** | 10.3% |
| `exogenous_expression` / Ca | 30,635 | 0.987 | 1.463 | **−48%** |
| `spatial_gradient` / **Nav** | 36,091 | 1.002 | 0.971 | 3.1% |
| `channel_blockade` / NaKATP | 16,931 | 0.892 | 0.857 | 3.9% |
| `channel_blockade` / Ca | 15,299 | 0.892 | 0.826 | 7.4% |
| `channel_blockade` / Cl | 16,956 | 0.710 | 0.667 | 6.1% |
| `spatial_gradient` / Cl | 29,732 | 0.577 | 0.590 | −2.2% |
| `spatial_gradient` / Ca | 30,802 | 0.532 | 0.572 | −7.5% |

Readings:

- **The spread is 6.5× across the table** (0.53 to 3.69 mV). Any single aggregate OOD number is
  an average over cases that differ by more than half an order of magnitude in difficulty.
- **`exogenous_expression` / Nav is by far the hardest case** at 3.3 mV, ~4× the in-distribution
  error. This is the family that combines spatial structure with input features at 4.0 when
  training saw only [0, 1] (§11.4) — it is the only genuinely hard OOD case in the set.
- **`spatial_gradient` / Ca and / Cl are easier than in-distribution data** (0.57, 0.59 vs 0.76).
  This confirms §5.7 quantitatively: those perturbations barely move Vmem, so they are not
  generalization tests at all. **Two thirds of the `spatial_gradient` split is not measuring
  generalization.**
- **The MPNN's largest wins are on `gj_blockade` (10.3%) and on Kir/K_leak blockade (14.4%,
  11.9%)**. `gj_blockade` is the family where topology should matter most — the graph is
  effectively severed — and it is one of the MPNN's better results. This is the second weak
  signal in favour of C2.
- **The MPNN is substantially *worse* on `exogenous_expression` / Ca (−48%)**, the one clear
  regression. With a single seed this could be noise, but it is large enough that multi-seed
  runs (Experiment D) are needed before any of this table is quoted.
### 10.6 Ablations (PRD §7.4)

All at seed 42, identical training path, early stopping on validation MAE. Rendered as figure 5.

| tag | arch | K | n_train | norm | params | test_id MAE | test_ood MAE | epochs |
|---|---|---|---|---|---|---|---|---|
| `abl_mlp_baseline` | mlp | — | 8000 | yes | 26,625 | 0.8119 | 1.2165 | 56 |
| `abl_depth_k2` | mpnn | 2 | 8000 | yes | 234,497 | 0.7792 | 1.1763 | 68 |
| `abl_depth_k4` | mpnn | 4 | 8000 | yes | 449,025 | **0.7474** | **1.0455** | 138 |
| `abl_depth_k6` | mpnn | 6 | 8000 | yes | 663,553 | 0.7923 | 1.1951 | 59 |
| `abl_depth_k8` | mpnn | 8 | 8000 | yes | 878,081 | 0.7589 | 1.0703 | 100 |
| `abl_size_1000` | mpnn | 6 | 1000 | yes | 663,553 | 0.8862 | 1.4060 | 87 |
| `abl_size_2000` | mpnn | 6 | 2000 | yes | 663,553 | 0.8169 | 1.3341 | 94 |
| `abl_size_4000` | mpnn | 6 | 4000 | yes | 663,553 | 0.7875 | 1.1973 | 94 |
| `abl_size_8000` | mpnn | 6 | 8000 | yes | 663,553 | 0.7666 | 1.1413 | 81 |
| `abl_no_normalize` | mpnn | 6 | 8000 | **no** | 663,553 | 0.7660 | 1.1055 | 80 |
| `abl_physics_loss` | mpnn | 6 | 8000 | yes | 663,553 | 0.7827 | 1.1259 | 64 |

**Read this table against the noise floor in §10.7: a per-run standard deviation of 0.0128 mV on
`test_id` and 0.0321 mV on `test_ood` for a *fixed* configuration and seed (n = 6).** Differences
below roughly two of those — 0.026 and 0.064 mV — are uninterpretable at n = 1 per cell, which
covers most of this table.

**Training-set size is the only ablation with a clearly resolvable effect.** 0.8862 → 0.8169 →
0.7875 → 0.7666 across 1K → 8K is monotonic and spans 0.12 mV, roughly 4× the noise floor.
Doubling data buys progressively less; the curve is flattening but has not saturated at 8K.

**Depth is not interpretable at n = 1 per depth.** K = 4 is nominally best on both splits and
K = 6 nominally worst, which is non-monotonic and therefore suspicious. On `test_ood` the K = 4 and
K = 8 results (1.0455, 1.0703) do sit below the entire observed K = 6 range (1.1413–1.1951) by more
than that range's width, which is the one depth signal that might survive replication. **The
correct reading today is that the depth sweep needs 3–5 replicates per depth before any claim, and
the PRD's "optimal propagation depth" question is unanswered.**

**Input normalization does nothing.** `abl_no_normalize` (0.7660) is indistinguishable from the
normalized runs (0.7615–0.7923). Given the network's first layer is a learned linear map followed
by LayerNorm, a fixed per-feature rescaling is largely absorbed — this is a mildly interesting
negative but not a surprising one.

**The physics auxiliary loss does nothing**, exactly as predicted in §7.3. 0.7827 sits inside the
fixed-config range. On spatially uniform tissues the junctional residual it penalizes is ~0
everywhere by construction, so the term supplies almost no gradient. This ablation is not a test of
the regularizer; it is another consequence of §11. It should be re-run after the training
distribution is fixed.

### 10.7 Reproducibility: the MPNN is nondeterministic, the MLP is not

The ablation grid accidentally re-ran the base configuration under two extra tags, which revealed
that runs sharing an identical configuration *and seed* do not agree. That was worth measuring
properly, so six replicates of MPNN / K = 6 / 8000 graphs / seed 42 now exist, plus a controlled
mechanism experiment.

#### The noise floor

| run | test_id MAE | test_ood MAE | epochs |
|---|---|---|---|
| `mpnn_seed42` | 0.7615 | 1.1642 | 91 |
| `abl_size_8000` | 0.7666 | 1.1413 | 81 |
| `abl_depth_k6` | 0.7923 | 1.1951 | 59 |
| `det_rep_1` | 0.7871 | 1.2285 | 48 |
| `det_rep_2` | 0.7840 | 1.1515 | 69 |
| `det_rep_3` | 0.7887 | 1.1649 | 59 |
| **mean ± sd (n = 6)** | **0.7800 ± 0.0128** | **1.1743 ± 0.0321** | 48–91 |

Epoch counts range from 48 to 91 for byte-identical inputs, so early stopping fires at genuinely
different points along genuinely different optimization trajectories.

#### The MLP reproduces bit-for-bit

`mlp_seed42` and `abl_mlp_baseline` are the same configuration run at different times and returned
identical values to every printed digit — 0.811936 / 1.216481, both stopping at epoch 56.

#### Mechanism: it is the scatter aggregation, on CPU as well as GPU

A controlled experiment — 500 training graphs, 10 epochs, patience disabled, two runs per
condition:

| condition | run 1 test_id | run 2 test_id | reproducible? |
|---|---|---|---|
| MPNN on **CUDA** | 1.106219 | 1.167929 | **no** |
| MPNN on **CPU** | 1.117110 | 1.056789 | **no** |
| MLP on CUDA | 1.204118 | 1.204118 | **yes, bit-identical** |

The MLP result rules out the obvious confounders: the dataloader shuffle, weight initialization and
the seeding of both `torch` and `numpy` are all shared with the MPNN and all reproduce exactly.
**The nondeterminism is specific to the MPNN, and it is present on CPU as well as CUDA.**

That rules out the first hypothesis, which was CUDA atomics alone. `MPNN.forward` aggregates
messages with `index_add`, whose reduction order is not fixed under parallel execution on *either*
backend — CUDA uses atomic floating-point adds, and PyTorch's CPU implementation parallelizes the
scatter under OpenMP. Floating-point addition is not associative, so the aggregation differs in the
last bits, and six message-passing layers across tens of epochs amplify that into divergent
training runs. `BaselineMLP` contains only dense linear layers, which are deterministic on both
backends.

#### What this means for the central comparison

**Correcting an earlier statement in this report:** an initial reading compared the *range* of three
accidental replicates (0.0308) against the mean MPNN−MLP difference (0.0305) and concluded the
effect was entirely inside the noise. That compared a range to a mean difference, which overstates
the noise. With n = 6 the per-run standard deviation is 0.0128 mV, and the two splits now separate:

| split | MPNN mean ± sd (n=6, seed 42) | MLP (deterministic) | difference | vs per-run sd |
|---|---|---|---|---|
| `test_id` | 0.7800 ± 0.0128 | 0.8119 | **0.0319** | **2.5×** (6.1× the SE of the mean) |
| `test_ood` | 1.1743 ± 0.0321 | 1.2165 | 0.0422 | 1.3× |

- **In distribution, the MPNN's advantage at seed 42 is real.** 0.0319 mV is 2.5× the per-run
  standard deviation and 6.1× the standard error of a 6-run mean. It is small — 3.9% — but it is
  not noise. It is, however, **invisible in any single run pair**, which is exactly what happened
  at seed 137 (§10.3), where one MPNN run and one MLP run tied at 0.780.
- **Out of distribution it is not established.** The effect (0.0422) is only 1.3× a per-run sd of
  0.0321, and the paired per-seed differences change sign (§10.3). OOD noise is 2.5× larger than
  in-distribution noise, which makes sense: the OOD split contains the extreme
  `exogenous_expression` inputs where small trajectory differences are most amplified.

So claim **C2 remains unsupported out of distribution**, and in distribution it is supported only
weakly, at an effect size of 0.03 mV — against a 10%-of-range accuracy target of ~8 mV. §11 remains
the explanation for why the margin is so small.

#### Consequences for the rest of this report

1. §10.3's seed study measured seed variance **plus** run nondeterminism. Its per-seed differences
   cannot be attributed to the seed, and each MPNN entry there is a single draw from a distribution
   with sd 0.0128 (test_id) / 0.0321 (test_ood).
2. §10.4 and §10.5 are single-run. Their large effects — the 3× boundary/interior ratio, the 6.5×
   OOD difficulty spread — are far above the noise and survive. The small ones, including the
   "MPNN's advantage is 2× larger at boundary cells" residual, do not.
3. In §10.6, any ablation difference below roughly 0.026 mV (test_id) or 0.064 mV (test_ood) —
   two per-run standard deviations — is uninterpretable. That covers normalization and the physics
   loss, and most of the depth sweep.

#### Remediation

In increasing order of cost: **report every model result as a mean over ≥ 3 replicates** (this is
now the project's standard); set `torch.use_deterministic_algorithms(True)` with
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, accepting reduced throughput on scatter ops; or replace
`index_add` with a deterministic segment-sum over a sorted edge list.

None of this was in the PRD. Its absence is why a 0.03 mV effect was first reported as a 21% win,
then 6.3%, then a null — **four successive corrections, every one in the direction that made the
result less impressive.** The pattern, not any individual number, is the methodological finding.

## 11. Analysis: the training distribution has no spatial structure

This is the central finding and it reframes every number in §10. It was found by asking a single
cheap question of the dataset — how much of the target variance is *within* a graph rather than
*between* graphs — after the baseline scored implausibly well.

### 11.1 The measurement

Measured directly over the generated records:

| split | per-cell density sd *within* a tissue | within-tissue Vmem sd | across-tissue Vmem sd |
|---|---|---|---|
| `train` | **0.000** | 1.33 mV (median 1.13, max 9.20) | 14.83 mV |
| `test_id` | **0.000** | 1.34 mV (median 1.09, max 8.15) | 13.92 mV |
| `test_ood` | 1.523 | 4.22 mV (median 1.75, max 42.12) | 16.46 mV |

`test_ood` decomposed by family (within-tissue Vmem sd):

| family | within-tissue Vmem sd |
|---|---|
| `exogenous_expression` | 8.10 mV |
| `gj_blockade` | 2.99 mV |
| `channel_blockade` | 1.57 mV |
| `spatial_gradient` (Nav only) | 10.58 mV |

### 11.2 What it means

**Every cell in every training tissue carries an identical channel-density vector.** The
sampler draws one density vector per configuration and tiles it across all cells
(`np.tile(per_channel, (n_cells, 1))`). Perturbations introduce per-cell variation; baselines
never do. Since train, val, and test_id are *entirely baseline*, the in-distribution problem is
degenerate in a specific way:

1. **≈ 99% of Vmem variance in the training set is between tissues, not within them**
   (14.83 vs 1.33 mV). Predicting the tissue mean from the shared density vector captures
   almost everything.

2. **The residual 1.33 mV is a boundary effect.** In a spatially uniform tissue there are no
   density gradients, so gap junctions have nothing to equalize — but cells at the tissue edge
   have fewer junctional neighbors and therefore couple less strongly to the bulk, producing a
   rim of slightly different potential. This is genuine gap-junction physics and it is the
   *only* graph-dependent signal present in the training data.

3. **This is why the MLP scores 0.970 mV on `test_id`.** Constrained to emit one value per
   tissue, its irreducible error is the within-tissue standard deviation — 1.33 mV in the
   training distribution, and its achieved 0.97 mV is consistent with predicting the tissue
   mean well and eating the rim as error.

4. **Gap-junction conductance is not merely under-used in-distribution; it is provably
   irrelevant there.** For a spatially uniform tissue every cell has the same transmembrane
   current at the same voltage, so the junctional term `Σ_j g_ij (V_i − V_j)` vanishes
   identically in the bulk regardless of `g_ij`. Coupling can only act at the boundary. The
   MLP's blindness to conductance costs it essentially nothing — not because the model is
   clever, but because the variable has almost no effect on the data as sampled.

### 11.3 The train/test structural mismatch

The model is trained exclusively on spatially uniform tissues and evaluated on spatially
heterogeneous ones — `exogenous_expression` at 8.10 mV within-tissue sd, Nav `spatial_gradient`
at 10.58 mV, against a training within-tissue sd of 1.33 mV. A network that never encountered
intra-tissue density variation during training received no gradient signal teaching it to
propagate information across the graph.

**Whatever OOD number the MPNN produces therefore measures extrapolation out of a degenerate
training regime, not the capability the Phase 1 claim is about.** The comparison is not a fair
test of the graph hypothesis. This must be stated in any paper; reporting an OOD MAE from this
setup as a generalization result would be misleading.

### 11.4 A compounding input-range shift

`exogenous_expression` sets 25% of cells to 4× the nominal channel maximum. Because the dataset
normalizer divides by the fixed `CHANNEL_MAXES`, those cells arrive with input features of
**4.0** when every training input lies in **[0, 1]**. That family therefore conflates two
distinct forms of extrapolation — spatial structure the model never saw, and input magnitudes
the model never saw — and cannot cleanly attribute error to either. It should be reported as a
combined stress test, with `spatial_gradient` (which stays within [0, 1]) as the clean spatial
generalization measurement.

### 11.5 Is this a bug or a finding?

Both, and the distinction matters for how it is written up.

It is a **design flaw** in the sampling strategy: nothing required baseline tissues to be
spatially uniform, and making them uniform removed the graph signal from 10,000 of the 12,000
records.

It is also a **genuine negative result** worth publishing as such: it demonstrates that a
graph-structured problem can be silently reduced to a pointwise one by a sampling choice, and
that a strong-looking R² (0.980) can be an artifact of that reduction rather than evidence of
learned physics. Benchmarks in scientific ML are vulnerable to exactly this failure, and it is
detectable with a single cheap measurement — the ratio of within-graph to across-graph target
variance — that is not standard practice. We recommend it be made standard practice.

---

## 12. Speed

### 12.1 The originally reported number was throughput, not latency

Every speedup claim made before Milestone 6 used a BETSE cost of **117.2 s per simulation**, a mean
over 13,800 runs during the generation campaign. That campaign ran **12 simulations concurrently**.
Under that load each individual simulation is slowed by contention, so 117.2 s is a
throughput-derived per-task figure, not single-simulation latency. PRD §7.3 asks for latency.

Measured serially and unloaded, the same simulator takes **~42 s** for tissues of 40–232 cells.
**The published denominator was inflated by roughly 2.5×, and the error flattered the model.** The
true speedup is smaller than previously stated in this report.

This is the same class of error as the unconverged-baseline and single-seed comparisons in §10.3:
a number that made the result look better, produced by measuring a convenient quantity rather than
the specified one.

### 12.2 Benchmark to protocol

Re-run per PRD §7.3: **100 configurations, executed one at a time**, median and interquartile
range, with CPU and GPU inference reported separately. The model-side timed region deliberately
includes graph construction and the host-to-device transfer, so it measures what a user would
actually wait for, not just the forward pass.

Deliberately **not** run concurrently with the ablation sweep, even though one is CPU-bound and the
other GPU-bound — concurrency is what corrupted the original figure.

**100 of 100 configurations completed, zero failures.** Tissue size median 150 cells (q1 85,
q3 275). Rendered as figure 4.

| | median | q1 | q3 | mean | min | max | speedup |
|---|---|---|---|---|---|---|---|
| BETSE, serial | **41.32 s** | 38.55 s | 48.81 s | 44.30 s | 33.73 s | 71.87 s | 1× |
| Model, CPU | **22.27 ms** | 18.29 ms | 30.50 ms | — | — | — | **1,856×** |
| Model, CUDA | **7.40 ms** | 6.51 ms | 8.31 ms | — | — | — | **5,582×** |

Claim **C4 is met** by a wide margin. The honest characterization is **three to four orders of
magnitude**, reported with measurement conditions attached rather than as a bare ratio.

Two caveats worth carrying into any writeup. The BETSE figure includes ~4 s of process startup per
simulation that a library-level integration would avoid, so it is a fair measure of *our* pipeline
rather than of the solver alone. And the comparison is not like-for-like in output: BETSE computes
a full time course of concentrations and voltages, of which we keep one field at one instant. The
model is not a faster simulator; it is a fast approximation of one scalar the simulator produces.

## 13. Planned experiments

| # | Experiment | Cost | What it decides |
|---|---|---|---|
| ~~**A**~~ | MPNN vs MLP by node degree, paired within graph | done | **Negative** (§10.4). The boundary/interior asymmetry appears identically in the graph-blind MLP, so it is a property of the data. Weak residual signal: the MPNN's advantage is ~2× larger at boundary cells. |
| ~~**B**~~ | OOD by family **and** perturbed channel | done | **Done** (§10.5). Difficulty spans 6.5×; two thirds of `spatial_gradient` is easier than in-distribution data. |
| **A2** | Test the mesh-geometry explanation for §10.4 — regress within-tissue Vmem deviation on cell area and neighbour count | free | Would establish the mechanism behind the interior/boundary asymmetry. |
| **C** | **Regenerate training data with intra-tissue spatial structure** | ≈ 27 h at 12 workers | The decisive fix for §11. Makes the graph informative in-distribution and turns MPNN-vs-MLP into a real test. |
| ~~**D**~~ | Multi-seed runs (42/137/256) for both architectures | done | **Done** (§10.3). OOD difference sign-flips across seeds; in-distribution difference is 2 wins and a tie. |
| **E** | Train with `physics_auxiliary_loss` enabled | ≈ 1 h | Only meaningful after C. |
| **F** | Message-passing depth sweep (K = 1, 2, 4, 6, 10) | ≈ 5 h | Measures the electrical coupling length; tests whether K = 6 is sufficient or excessive. |
| **G** | Experimental validation against published *Xenopus* Vmem measurements | unscoped | The only test of whether BETSE itself is right. |

Experiment C is the one that matters. Experiments A and B are now complete and both point the
same way: the current comparison is not measuring what it was supposed to measure, and no amount
of further analysis of these splits will change that. Everything else refines a comparison whose
signal is bounded at ~1.33 mV of within-tissue variance.

---

## 14. Limitations

1. **The target is Vmem after 5 s of equilibration, not a steady state.** No steady state
   exists in this system (§4). The `.npz` key name `vmem_steady_state` is a compatibility
   artifact.
2. **Two of eight input features are identically zero** across the entire dataset (§2.3). The
   effective input dimension is six.
3. **The training distribution contains no intra-tissue spatial variation** (§11). This is the
   most serious limitation and it bounds what any current Phase 1 result can claim.
4. **Cl and Ca perturbations are not meaningful generalization tests** — they move Vmem by
   ~0.5 mV, under 3% of the accuracy target (§5.7).
5. **`gj_blockade` is a near-blockade**, floored at 1% of open conductance rather than zero
   (§5.5).
6. **`exogenous_expression` confounds spatial and input-magnitude extrapolation** (§11.4).
7. **`Nav` denotes background sodium permeability, not voltage-gated Nav density** (§3.2). The
   feature name does not mean what it appears to mean.
8. **Spatial perturbations are restricted to three of six channels** by simulator interface
   limits, not by design (§5.2).
9. **No experimental validation.** The model is validated against a simulator; the simulator's
   fidelity to biology is assumed, not tested (Experiment G).
10. **Single seed.** *n* = 1 for every number in §10. No error bars yet.
11. **Per-cell metric weighting** means large tissues dominate the reported averages (§9).
12. **The mechanism behind the interior/boundary error asymmetry is not established** (§10.4).
    It is a property of the data, but which property is untested.
13. **One OOD cell shows a large regression** — `exogenous_expression` / Ca, where the MPNN is
    48% worse than the MLP (§10.5). Unexplained, and with *n* = 1 it cannot be distinguished
    from noise.

---

# Part II — The director–executor architecture

## 15. What this part is about

The entire NEXUS codebase was written by **`qwen2.5-coder:7b`**, a 7-billion-parameter local
model served by Ollama. A **Claude Opus "director"** process wrote **no code at any point** —
not a function, not a fix, not a one-line patch. The director decomposed tasks, wrote
natural-language specifications, ran the test suite, read raw output, and fed failures back.

This is not a productivity story. It is a structured experiment with its own hypothesis:

> A small model's coding competence is not fixed by its weights. It is a function of the
> *environment structure* the director provides. By varying that structure systematically and
> measuring where convergence fails, the capability boundary can be located.

The test suite — 92 tests, written before any implementation and never modified — is the
objective function. It is what makes this an experiment rather than an anecdote.

## 16. Setup

```
Mac (Claude Opus director)
   │  SSH over Tailscale
   ▼
Windows machine (RTX 4050, 6 GB)
   ├── Ollama serving qwen2.5-coder:7b
   ├── nexus-phase1/          all deliverable code
   ├── nexus-phase1/tests/    the frozen acceptance suite
   └── nexus-phase1/logs/     director notes
```

The executor is queried over Ollama's HTTP API at temperature 0.1 with a fixed seed and an
8192-token context. Low temperature is deliberate: the executor is not being asked to be
creative, it is being asked to transcribe a specification into syntax deterministically.

Deployment is a heredoc with a quoted delimiter (`<< 'PYEOF'`) so the shell cannot expand
Python source in transit — a failure mode that silently corrupts `$` and backtick characters.

## 17. The six architectural properties

The director operates under six explicit constraints, adopted before the build began.

**P1 — State reflection.** Every instruction opens with a factual status line
(`System state: 92/92 tests passing. Current file: scripts/train.py.`). Corrections include the
file verbatim, the test verbatim, and the raw pytest output verbatim — and **nothing else**. No
diagnosis, no suggested fix, no evaluative language. The executor perceives what *is* and
determines what to do. The director assembles context; it does not prescribe.

The single exception: after three identical failures on the same test, one factual hint about
the test's expectation is permitted.

**P2 — Tiered engagement.** Tasks are tiered by structural complexity — Tier 1 pure functions,
Tier 2 stateful classes, Tier 3 classes composing multiple modules. Promotion requires
demonstrated first-attempt passes at the current tier.

**P3 — Mode-dependent topology.** Context is scoped to the current task only. Writing
`losses.py` means seeing function signatures and the torch import — not the model, not the
trainer, not the dataset. **The full specification is never sent to the executor.** A 7B model
given a 40-page PRD and asked to "build the system" produces incoherent output; given one file
with exact signatures, it produces correct output.

**P4 — Consequence observation.** Every instruction opens with one factual sentence about the
previous one: *"Your last file (nexus/training/trainer.py) passed all 10 tests."* No praise, no
criticism. This creates a temporal feedback loop across otherwise stateless API calls.

**P5 — Scaffold degradation.** Support attenuates as coherence is demonstrated. Full scaffold
(exact signature, exact imports, line-by-line behavior) → partial (name, one-sentence
description, which test class) → minimal (file path and test names only). Promotion after three
consecutive first-attempt passes; **failure at a degraded tier forces escalation back to full
scaffold**. That escalation is the self-falsification test: it reveals whether apparent
competence was scaffold-dependent.

**P6 — Coherence verification.** The test suite is a health signal, not a score to maximize. If
previously-passing tests start failing, forward progress stops until the regression is
diagnosed.

## 18. Results: where the capability boundary is

**At full scaffold, the 7B succeeded on every file in the system**, including all three Tier 3
files (`mpnn.py`, `dataset.py`, `trainer.py`). A 664K-parameter graph network with residual
message passing, degree-normalized aggregation, and a fixed output affine map was produced from
a natural-language specification by a 7B model, and it passes 24 architecture tests including
permutation equivariance and edge sensitivity.

**At partial scaffold, it failed on `trainer.py`** — with five simultaneous defects:

1. `UnboundLocalError` on the early-stopping counter (used before initialization)
2. an undefined `history` reference inside `save_checkpoint`
3. an unconditional three-argument model call, breaking the MLP path
4. a removed PyG import
5. a wrong optimizer/scheduler/early-stopping combination

Escalating that task back to full scaffold produced a correct file immediately.

**The boundary is therefore sharp and it is located at the interaction of task tier and
scaffold tier**: "class composing multiple modules" is *within* reach at full scaffold and
*outside* it at partial scaffold. The executor's competence is not a property of the model
alone. It is a property of the (model, scaffold) pair.

This is the experiment's main methodological result, and it has a practical corollary: the
useful question about a small model is not "can it write X" but "what is the minimum
specification density at which it can write X."

## 19. Failure taxonomy

Cataloguing the executor's failure modes, since they are systematic rather than random.

**F1 — Insertion damage.** Instructing the executor to *insert* code into a long existing file
causes it to delete adjacent unrelated lines. In one instance it removed six consecutive lines
(`positions`, `nn`, `keep`, `edge_index`, `gjopen`, `conductances`) while adding one.
*Mitigation:* never phrase an edit as an insertion. State the entire replacement region
verbatim and request a full-file rewrite.

**F2 — Dropped-line elision.** Given a code block embedded in prose, the executor sometimes
omits exactly one line. Observed twice: writing `inspect.signature(...)` while omitting
`import inspect`; and keeping `params = inspect.signature(...)` while dropping the
`self.needs_graph = ...` line that consumed it. *Mitigation:* for multi-line bodies, state the
complete block and its line count.

**F3 — Over-literal interpretation.** "Sorted by the `config_id` field" produced
`key=lambda x: int(x["config_id"])`, which raises `ValueError` on IDs like `base_000011`.
The executor inferred a numeric sort from a field named `id`. *Mitigation:* say "sort the
string."

**F4 — Destructive list operations.** Asked to configure tissue profiles, the executor wrote
`del prof_list[:]` before appending, deleting BETSE's shipped `Spot` profile and producing
`KeyError: 'Spot'`. *Mitigation:* specify append-vs-replace explicitly whenever mutating a
structure the executor cannot see.

**F5 — Type confusion in arithmetic.** `total_wall_clock_s / (completed − done)` where `done`
is a `set`, raising `TypeError` at the tenth completion. This one survived into a long-running
job and was caught only because the job was tested against partial data rather than left to
fail at hour 36.

**F6 — Test code in a source module.** A `TestConfigSampler` class was written into
`nexus/data/config_sampler.py`. Functionally inert — pytest's `testpaths` is `tests/` — but
wrong, and undetectable by the test suite. Caught by reading the source during report writing.

**F7 — Invented conventions in place of stated ones.** Told to read files via
`sorted(glob.glob(os.path.join(args.data, "test_ood", "*.npz")))`, the executor instead wrote
`os.path.join(args.data, "test_ood", f"{i}.npz")` — substituting a plausible naming convention
for the one specified. *Mitigation:* the same as F3; but note the executor had the correct line
verbatim in its prompt and did not use it, which is a stronger failure than mis-inference.

**F8 — Block relocation.** Told to add a statement inside a loop immediately after an existing
line, the executor placed it after the loop instead and rewired it to iterate over the wrong
collection (`report["splits"].items()`, which holds metric dicts, rather than the per-split
prediction arrays). The code parsed and would have raised a `KeyError` at runtime.
*Mitigation:* the F1 mitigation generalizes — specify the entire enclosing function verbatim
rather than describing where a line goes. Doing so resolved both F7 and F8 in one round.

The unifying pattern: **the executor's failures are local and syntactic, not architectural.**
It does not misunderstand what an MPNN is. It drops a line, deletes a neighbor, over-infers a
type, or relocates a block. These are exactly the failure modes a test suite catches — which is
why the architecture works, and why the two scripts *without* test coverage
(`scripts/train.py`, `scripts/evaluate.py`) needed the most correction rounds of any files in
the project. **Test coverage and correction cost are inversely related**, and that relationship
is the strongest practical argument for writing the suite first.

A quantified version, counting first-attempt outcomes on the files written in this session:

| File | Test-covered | Correction rounds |
|---|---|---|
| `nexus/training/config.py` | yes | 0 |
| `nexus/training/trainer.py` (device change) | yes | 1 |
| `nexus/data/config_sampler.py` (cleanup) | yes | 0 |
| `scripts/train.py` | **no** | 1 |
| `scripts/evaluate.py` | **no** | **2** |

## 20. Director-error analysis

Counting honestly, **the majority of correction rounds traced to defects in the director's
specification, not the executor's implementation.**

| Director error | Consequence |
|---|---|
| Hardcoded `sim/config.yaml` | BETSE on Windows rejects forward slashes; a full debug cycle |
| Mapped K⁺ to `Dm_K` | Measurement later showed the parameter is inert; mapping rewritten |
| "sorted by the `config_id` field" | Invited the `int()` cast of F3 |
| Quoted `max_dvmem ≈ 1e-5` as convergence | Wrong simulation phase; repeated twice before catching |
| Launched a 36-hour campaign without profiling BETSE's time settings | Unrecoverable wall clock |
| Asked for an *insertion* into a long file | Triggered F1 |

This ratio is the most useful finding in Part II for anyone building a similar system. The
bottleneck was not the executor's capability. It was the **precision of the natural-language
interface**, and specifically the director's tendency to write specifications containing
implicit assumptions that a 7B model cannot supply and a larger model would have silently
patched. The small model is a *specification linter*: it fails loudly on ambiguity that a
stronger model would paper over.

## 21. Infrastructure findings

**21.1 — Ephemeral port exhaustion starved the code generator.** Ollama repeatedly failed with
`timed out waiting for llama-server to start`. Root cause: 25,010 sockets stuck in `TIME_WAIT`
against Windows' default 16,384-port dynamic range (49152–65535), roughly 13,000 of them from
`joblib`/`loky`'s per-task worker IPC. The data generation campaign was consuming every
ephemeral port on the machine and the code generator could not open a listening socket.

Fix: `netsh int ipv4 set dynamicport tcp start=20000 num=45535`
(revert: `start=49152 num=16384`).

The general lesson: a parallel data pipeline and a local model server contend for a resource
neither of them names in its documentation.

**21.2 — GPU contention is total on a 6 GB card.** The MPNN occupies 5.77 GB of 6.14 GB during
training. Ollama cannot load a 7B model alongside it, so **the executor is unavailable for the
entire duration of any training run.** A `config_sampler.py` cleanup request issued during
training timed out after 10 minutes with no response. On a single-GPU rig, code generation and
model training are strictly serial activities, and the director must schedule around that.

**21.3 — SSH sessions drop under sustained load.** Long-running jobs must be launched detached
via `Win32_Process.Create` rather than as SSH children, and poller sessions must not run
concurrently with a long session.

**21.4 — Test the finalizer against partial data.** F5 was caught because the finalization
script was run against a partially-complete dataset mid-campaign rather than trusted to work at
the end. For any job measured in tens of hours, dry-running the terminal step against
incomplete inputs is worth the interruption.

## 22. Honest assessment of the architecture

**What worked.** A 7B model wrote a 22-file scientific codebase that passes 92 tests including
real-simulator integration. The scaffold-degradation protocol located a reproducible capability
boundary. The test suite as an immutable objective function made every claim checkable. Failure
modes proved systematic and mitigable.

**What did not.** Iteration is slow — each correction is a full-file regeneration, and on a
6 GB GPU it cannot overlap with training. The director wrote more defective specifications than
the executor wrote defective code. Scaffold degradation was attempted on a small number of
tasks; the promotion thresholds (three consecutive first-attempt passes) mean the sample size
behind the "partial scaffold fails at Tier 3" claim is **one task**. That is suggestive, not
conclusive, and a paper must say so.

**What is genuinely novel.** The director/executor split is not new. What is less common is
treating the split as an *experiment* with a frozen objective function and a pre-registered
protocol for varying support structure — and then reporting that the majority of failures were
the director's. That accounting is only possible because the test suite was written first and
never touched.

## 23. Threats to validity (Part II)

1. **n = 1 on the key claim.** One task failed at partial scaffold. Locating a boundary requires
   more tasks at each scaffold tier.
2. **No control condition.** The 7B was never asked to build the system without the six
   properties, so their contribution is not isolated.
3. **Director-error counting is retrospective and self-reported**, and therefore subject to
   hindsight bias in both directions.
4. **The test suite constrains the design space heavily.** With exact signatures fixed in
   advance, "write the file" is closer to transcription than to software design. This makes the
   task easier than open-ended development, and the results should not be read as a claim about
   the latter.

---

## 24. Changelog

**2026-08-30** — First training runs on real BETSE data, and the first negative results.
- Added GPU support to `TrainingConfig` / `Trainer` (`device` field, appended last to preserve
  all existing call sites). Full suite re-verified green at **92/92** including 5 BETSE
  integration tests, 498.6 s.
- `scripts/train.py` and `scripts/evaluate.py` written.
- **Converged comparison (§10.3): MPNN `test_id` MAE 0.761 mV vs MLP 0.812 mV — a 6.3% margin
  for 25× the parameters. Claim C2 is not supported.** An earlier unconverged 2-epoch MLP made
  the margin look like 21%; that comparison is retained in §10.2 as a caution.
- **Measured within-tissue vs across-tissue Vmem variance and identified the uniform-density
  training distribution as the cause** (§11): per-cell density sd is exactly 0.000 in every
  training tissue, and ~99% of Vmem variance is between tissues rather than within them.
- **Degree control (§10.4) came out negative.** Interior cells are ~3× harder than boundary
  cells for both models, paired within graph — but the graph-blind MLP shows the same asymmetry
  at the same magnitude, so it is a property of the data, not of message passing. Residual weak
  signal: the MPNN's advantage is ~2× larger at boundary cells.
- **OOD cross-tabulated by family and perturbed channel (§10.5).** Difficulty spans 6.5×;
  `spatial_gradient` / Ca and / Cl are *easier* than in-distribution data, confirming that two
  thirds of that split does not test generalization. Hardest case is
  `exogenous_expression` / Nav at 3.3 mV.
- Identified the `exogenous_expression` input-range shift (features at 4.0 vs training [0, 1]).
- Removed `TestConfigSampler` from `nexus/data/config_sampler.py` (F6); suite re-verified.
- New executor failure modes F7 (invented conventions) and F8 (block relocation), and the
  observation that the two files without test coverage required the most correction rounds
  (§19).
- Documented GPU contention between training and the local executor (§21.2).
- Report restructured into Part I (science) and Part II (methodology).

**Earlier** — Dataset generation campaign.
- 13,800 simulations, **zero failures**, 117.2 s each under 12-way load, ≈ 39 h. (Serial latency
  later measured at ~42 s — see §12.1.)
- 12,000 records finalized into 8000 / 1000 / 1000 / 2000 splits, verified disjoint.
- Per-cell tissue-profile mechanism developed and verified in production
  (`corr(x, density) = 0.997`; baseline path bit-identical).
- OOD heterogeneity characterized: only Nav perturbations move Vmem appreciably.
- Convergence criterion found unsatisfiable; target renamed to "Vmem after 5 s equilibration."
- Physics mapping derived by parameter sweep rather than documentation.
- Ephemeral port exhaustion diagnosed and fixed.
