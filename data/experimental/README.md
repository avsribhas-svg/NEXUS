# Experimental validation data — curation protocol and mapping assumptions

**Read this before using anything in this directory.** The mapping from a published experiment to
a model input is the scientifically load-bearing step in Milestone 7, and it is approximate. This
file states every assumption so a reader can judge the validation rather than take it on trust.

PRD reference: §4.2 (curation protocol), §7.5 figure 7, §10 Milestone 7.

---

## 1. The problem

The model consumes a graph: per-cell channel densities `x` of shape `(n_cells, 8)` normalized by
`CHANNEL_MAXES`, a gap-junction `edge_index`, and per-edge conductances. A published experiment
gives none of these. It gives a tissue, a reagent, and a voltage.

Bridging that gap requires three assumptions, each of which can be wrong independently:

| # | Assumption | Risk |
|---|---|---|
| A1 | The experimental tissue can be represented by *some* point in our 11-D sampled parameter space | Our space was designed around BETSE's knobs, not around biology |
| A2 | The reagent's action maps onto a change in one of our **six mapped** channels | Most published bioelectric reagents target channels we do not have (§3) |
| A3 | BETSE's physics is correct for the tissue in question | BETSE's own validation is ~1.5 mV on one preparation (§4) |

A validation that ignores A1–A3 and reports a bare MAE would be misleading. This document exists so
that the reported number carries its assumptions with it.

---

## 2. Validation design: differential, not absolute

**We do not attempt to predict absolute Vmem from a literature description of a tissue.** Doing so
would require inventing a channel-density vector, and the invented vector would determine the
answer. That is not a test.

Instead the primary protocol is **differential**:

1. Take a measured control potential `V0` and a measured perturbed potential `V1` from the same
   experiment, giving an observed shift `ΔV_exp = V1 − V0`.
2. Select from the generated dataset the ensemble of baseline configurations whose mean Vmem lies
   within a tolerance of `V0`. This is a *matched-baseline ensemble*: many different channel
   vectors produce the same resting potential, and we keep all of them rather than choosing one.
3. Apply the perturbation to every member of the ensemble using the same operation the dataset's
   own perturbation families use — for a channel blockade, set that channel's column to zero.
4. Run the model on each perturbed member. This gives a **distribution** of predicted shifts
   `ΔV_pred`, not a point estimate.
5. Compare the distribution of `ΔV_pred` against `ΔV_exp`.

Why this is the right call:

- It never requires knowing the absolute channel densities of a real tissue, only that *some*
  configuration reproduces its resting potential.
- The spread of the ensemble is an honest expression of the degeneracy of the inverse problem —
  many channel vectors give the same Vmem, and they need not respond identically to a perturbation.
  A wide predicted distribution is itself a finding.
- It matches what the experiments actually report. Most bioelectric papers report a *shift* under
  a reagent, often via voltage dye, far more reliably than a calibrated absolute value.

A secondary, weaker check is **range overlap**: does the model's predicted Vmem distribution cover
the range of resting potentials reported for non-excitable cells? This tests that the model is in
the right regime, nothing more.

---

## 3. The blocking limitation: the literature perturbs channels we do not have

Our 8-channel vector holds `HKATP` and `VATP` at exactly zero because BETSE exposes no equivalent
(see `DEVIATIONS.md` D1). Six channels are mapped: `Nav`, `Kir`, `K_leak`, `Ca`, `Cl`, `NaKATP`.

The canonical bioelectric-patterning experiments named in PRD §4.2 target, respectively:

| Experiment | Molecular target | In our channel set? |
|---|---|---|
| Beane et al. 2011, planarian head regeneration | **H⁺/K⁺-ATPase** (SCH-28080) | **No** — held at zero |
| Pai et al. 2018, nicotine teratogenesis | **nicotinic ACh receptor** | **No** |
| Pai et al. 2018, HCN2 rescue | **HCN2** hyperpolarization-activated channel | **No** |
| Adams & Levin 2012, craniofacial patterning | V-ATPase and others | **No** — held at zero |

**Three of the four PRD-named primary sources perturb a channel this parameterization cannot
represent.** This is not a curation failure; it is a consequence of choosing the parameter space
around BETSE's configuration surface rather than around the experiments that would validate it.
It is the single most important finding of Milestone 7 and it should be stated as such.

What remains testable is any experiment perturbing `Kir`, `K_leak`, `Nav`, `Ca`, `Cl`, `NaKATP`, or
gap-junction coupling — e.g. Ba²⁺ block of Kir, ouabain block of Na⁺/K⁺-ATPase, or octanol /
carbenoxolone uncoupling.

---

## 4. The accuracy ceiling

BETSE's own published validation against experiment is a single comparison: *Xenopus* oocytes in
Ringer's solution, experimental **−39.1 mV** against BETSE's predicted **−37.6 mV**, using ion
permeabilities and concentrations from Costa et al. 1989 — reported as **< 10% difference**
(Pietak & Levin 2016, Table 2).

Our model is trained to approximate BETSE. **It therefore cannot be more accurate against
experiment than BETSE is**, and its error against experiment is the sum of BETSE's error against
biology and our model's error against BETSE. The latter is ~0.8 mV (§10.3 of the research report).
The former is ~1.5 mV on the one preparation where it has been checked, and unknown elsewhere.

This also explains PRD §2's 10%-of-range accuracy target: it is inherited from BETSE's own
validated accuracy, not chosen independently.

---

## 5. Inclusion criteria

A record enters `data/experimental/*.json` only if:

1. The citation is real and was independently verified to exist.
2. The value is attributable to a stated condition in that source.
3. `is_absolute` is set truthfully. Voltage-sensitive dyes (DiBAC4(3), CC2-DMPE/DiSBAC2(3)) report
   **relative** intensity unless the paper states calibration against electrode recordings. Where a
   paper calibrates its dye against whole-cell recordings, that is noted explicitly per record.
4. Values taken from a paper's *model* output rather than its *measurements* are excluded from
   ground truth, or flagged `source_is_model: true` and excluded from the accuracy computation.
   This matters: at least one PRD-named source reports simulated and measured voltages in adjacent
   sentences.

**What we could not do.** PRD §4.2 specifies extracting Vmem from published colormap figures using
WebPlotDigitizer or equivalent. That requires reading pixel values out of figure images and is not
available here. Every value in this directory is a number stated in text or in a table, never one
digitized from a heatmap. This narrows coverage substantially, because much of the Xenopus
bioelectric literature publishes voltage as a colormap and not as tabulated numbers.

---

## 6. Record schema

```json
{
  "id": "kir_blockade_barium_smooth_muscle",
  "organism": "...", "tissue": "...",
  "condition_control": "...", "condition_perturbed": "...",
  "perturbation_family": "channel_blockade",
  "mapped_channel": "Kir",
  "vmem_control_mv": -60.0, "vmem_perturbed_mv": -50.0,
  "delta_vmem_mv": 10.0, "uncertainty_mv": 0.0,
  "method": "sharp microelectrode",
  "is_absolute": true, "source_is_model": false,
  "citation": "...", "doi_or_url": "...",
  "verification": "CONFIRMED | CORRECTED | UNVERIFIABLE",
  "mapping_assumption": "prose statement of how the reagent maps onto a channel-density change"
}
```

`mapping_assumption` is mandatory and is never boilerplate. It is the field a reviewer should read
first.

---

## 7. Results (added after the validation run)

Curation yielded **30 candidate records; 29 confirmed, 1 unverifiable, 0 fabricated**, all absolute
millivolt measurements from electrode recordings. Of these, **6 form control/perturbed pairs whose
perturbation this parameterization can represent**, plus 18 baseline anchors and 6 documented
exclusions.

Experimental Vmem range across the anchors is 57.8 mV, so PRD §2's threshold is **5.78 mV**.

| stratum | n | MAE | verdict |
|---|---|---|---|
| Channel blockade, representable channel | 2 | **5.35 mV** | **meets** |
| Gap-junction blockade | 3 | 9.80 mV | fails |
| Channel blockade, knowingly unrepresentable tissue | 1 | 34.65 mV | fails, as predicted |
| **All** | 6 | **12.46 mV** | **fails** |

### The gap-junction result

All three gap-junction records predict ≈ 0. Complete uncoupling, measured at +18.8 mV, is predicted
at **+0.017 mV**.

This is not a poor prediction, it is a categorical one, and it was **predicted in advance**. Every
training tissue in this dataset is spatially uniform, so the junctional term vanishes in the bulk
for any conductance (research report §11, `DEVIATIONS.md` D8). The model correctly learned that
gap-junction conductance does not affect its training targets. Confronted with real tissue where
uncoupling shifts Vmem by 19 mV, it predicts nothing.

A dataset diagnostic and an independent experimental failure pointing at the same cause is stronger
evidence than either alone.

### The deliberate counterexample fired

`barium_locust_malpighian` was included specifically because the model should fail it — same
reagent and nominal target as `barium_frog_kidney`, opposite measured sign, because insect
Malpighian tubules are driven by an apical V-ATPase held at zero (D1). The expected failure was
written into its `mapping_assumption` field before the model was run. Predicted +16.65 mV against a
measured −18.00 mV: wrong by 34.65 mV and wrong in sign.

Including a case you expect to fail, and recording that expectation first, is cheap insurance
against a mapping protocol flexible enough to absorb any result.

### Reproduce

```bash
python scripts/validate_experimental.py \
    --arch mpnn --checkpoint outputs/mpnn_seed42/final.pt --device cuda
```

Outputs `outputs/experimental_validation.json` and `outputs/figures/fig7_experimental_validation.png`.
