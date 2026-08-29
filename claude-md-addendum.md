# CLAUDE.md Addendum — Six Architectural Properties

Paste this into the CLAUDE.md after the Rules section and before Success Criteria.
These override any conflicting instructions above.

---

## Architectural Properties

The following six properties govern how you (Opus) structure the 7B's environment.
These are not guidelines. They are the architecture. The 7B converges because of
these structural properties, not because of its weights.

### Property 1: State Reflection

The 7B perceives current system state as fact, not evaluation.

**In practice**: Every instruction to the 7B includes a factual status line:

```
System state: 17/80 tests passing. Current file: nexus/model/mpnn.py.
```

When correcting code, you MUST include:
- The file the 7B wrote (verbatim, full contents)
- The test that was run (verbatim, the test function)
- The raw pytest output (verbatim, including traceback)

You MUST NOT include:
- Your interpretation of why it failed
- Your suggestion for how to fix it
- Any evaluative language ("this is wrong", "you made an error")

**Correction prompt template**:
```
System state: 17/80 tests passing. Current file: nexus/model/mpnn.py.

Here is the file you produced:
---
[full file contents]
---

Here is the test:
---
[test function from the test file]
---

Here is the output when the test ran:
---
[raw pytest output]
---

Write the corrected version of this file.
```

The 7B sees what is. It determines what to do. You assemble context; you do not prescribe fixes.

The ONE exception: if the 7B produces the same incorrect output 3 times in a row
on the same test (identical failure mode), you may add ONE factual hint:
"Note: the test expects output shape (n_nodes,), not (n_nodes, 1)."
This is still a fact about the test, not a directive.


### Property 2: Tiered Engagement

Capabilities gate on demonstrated readiness. Harder tasks are irrelevant until
easier tasks are demonstrated.

**Task tiers**:

Tier 1 (pure functions, no class structure, no cross-module dependencies):
- nexus/model/losses.py
- nexus/evaluation/metrics.py
- nexus/data/validation.py
- nexus/training/config.py
- nexus/evaluation/generalization.py

Tier 2 (classes with internal state, single-module scope):
- nexus/model/baseline.py
- nexus/data/config_sampler.py
- nexus/evaluation/figures.py

Tier 3 (classes composing multiple modules, full data flow):
- nexus/model/mpnn.py
- nexus/data/dataset.py
- nexus/training/trainer.py

**Promotion rules**:
- Start at Tier 1.
- Promote to Tier 2 after 3 Tier 1 tasks pass on first attempt (no correction needed).
- Promote to Tier 3 after 2 Tier 2 tasks pass on first attempt.
- If a task at the current tier fails after 5 correction rounds, do NOT promote.
  Stay at the current tier. Log the ceiling.
- If promoted and then failing at the new tier, demote back. The promotion was premature.

**In the director log**, record the 7B's current tier and its promotion/demotion history.


### Property 3: Mode-Dependent Topology

What the 7B can see depends on what it is working on. You scope the context
included in each instruction to ONLY what is relevant to that task.

**Context scoping rules**:

| Task | What the 7B sees | What the 7B does NOT see |
|---|---|---|
| losses.py | Function signatures, torch import | Model architecture, trainer, dataset |
| baseline.py | losses.py interface, Data object shape | MPNN internals, trainer, evaluation |
| mpnn.py | losses.py interface, Data object shape, baseline.py interface (for comparison) | Trainer internals, evaluation, dataset loading |
| config.py | Field names and types only | Everything else |
| trainer.py | Model forward signature, loss functions, config fields, DataLoader interface | Dataset construction, evaluation, figures |
| dataset.py | Data object required attributes (from tests), npz file schema | Model, trainer, evaluation |
| metrics.py | numpy, function signatures | Everything else |
| figures.py | matplotlib, metric functions interface | Model, trainer, dataset |

**Never** include the full PRD in a 7B instruction. The PRD is YOUR reference.
The 7B sees only the interface surface of its current task.


### Property 4: Consequence Observation

The 7B perceives actual effects of its actions as natural feedback.

This is implemented by Property 1's correction template. But there is an
additional rule:

**After every task (pass or fail)**, the NEXT instruction to the 7B begins with:

```
Your last file ([filename]) [passed all N tests / failed N of M tests].
```

This is one factual sentence. No elaboration, no praise, no criticism.
The 7B perceives the consequence of its previous action before beginning the
next one. This creates a temporal feedback loop even across stateless API calls.

If the last task passed, this is confirmation. If it failed and you moved on
(skip rule), this is information about the system's state that the 7B carries
into the next task.


### Property 5: Scaffold Degradation

Support structures attenuate as behavioral coherence is demonstrated.

**Instruction detail tiers**:

- **Full scaffold** (starting tier): Exact function/class signature, exact
  imports, exact behavior description line by line, the test function it must
  pass, and (for complex tasks) example input → output.

- **Partial scaffold** (earned after 3 consecutive first-attempt passes):
  Function/class name, one-sentence behavior description, which test file and
  test class to pass. No exact signature. No imports. The 7B infers from the
  test expectations.

- **Minimal scaffold** (earned after 3 more consecutive first-attempt passes at
  partial): The file path and which tests to pass. Nothing else.

**Degradation rules**:
- Start at full scaffold.
- Promote after 3 consecutive first-attempt passes at current tier.
- If a task FAILS at partial or minimal scaffold: escalate back to full scaffold
  for that specific task. This is the self-falsification test. Log:
  "Scaffold degradation test: 7B failed [task] at [partial/minimal] scaffold.
  Re-attempting at full scaffold. Previous convergence was scaffold-dependent."
- After re-passing at full scaffold, the scaffold tier resets. Earn partial again.

**In the director log**, record the current scaffold tier and every
degradation/escalation event. This data is scientifically interesting:
it characterizes where the 7B's capability boundary actually is.


### Property 6: Coherence Verification

The environment provides ongoing signal about whether agent behavior is coherent
with system health, as information rather than a score to maximize.

**Implementation**: The test suite is the coherence signal. After every test run,
Opus logs the full test state:

```
## Coherence State
Tests passing: 34/80 (42.5%)
Last 5 tasks: pass, pass, fail, pass, pass
Current tier: 2 | Scaffold: partial
Failing tests: [list the failing test names]
```

This goes in the director log after every task. It is NOT sent to the 7B as a
score. The 7B receives only the one-line factual consequence (Property 4).
The coherence state is for YOU (Opus) and for the user.

**Coherence-based decisions**: If the coherence state is degrading (tests that
previously passed are now failing after new code was added), STOP forward
progress. Diagnose which new file broke which old test. Fix the regression
before adding new code. Coherence verification is about system health, not
forward velocity.

---

## Property Implementation Checklist

Before sending any instruction to the 7B, verify:

- [ ] Status line included? (Property 1)
- [ ] Context scoped to current task only? (Property 3)
- [ ] Consequence of last task stated? (Property 4)
- [ ] Scaffold tier appropriate for 7B's demonstrated capability? (Property 5)
- [ ] If correction: raw code + raw test + raw output included, no evaluation? (Property 1)
- [ ] Current tier appropriate for task complexity? (Property 2)

Before logging a milestone, verify:

- [ ] Coherence state logged? (Property 6)
- [ ] Any regressions addressed before proceeding? (Property 6)
- [ ] Scaffold tier transitions logged? (Property 5)
- [ ] Tier promotions/demotions logged? (Property 2)
