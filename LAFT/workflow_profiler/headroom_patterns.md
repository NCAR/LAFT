# Headroom analysis patterns — signature → hypothesis → candidate fix

Reference for Step 7 of `PROFILE_WORKFLOW.md`. This file is
codebase-agnostic and LLM-agnostic: it maps measurement *signatures* to
structural *hypotheses* and candidate fixes. It never gates —
`production_ready` comes only from `diagnose.py`. Use it after the nsys
pass, with the cost ladder built and the staged code open: every hypothesis
below must be confirmed against the code (file:line) before it becomes a
recommendation.

## The signatures

| Signature (from the ladder / nsys / trace) | Likely cause | Confirm in the code | Candidate fix |
|---|---|---|---|
| Device-idle fraction > 50% of the call | Host-bound dispatch: the host cannot feed the device | Count jitted entry points called per invocation and arguments per jitted call — many small jit calls, or very wide argument lists (pytree flatten cost), or host Python between stages | Coarsen jit boundaries: one `@jax.jit` around the numeric pipeline (flags → `static_argnames`), or merge stages into a few super-stages; watch compile time |
| Many D2D memcpys per call | Jit-boundary materialization: intermediates forced to device buffers between separately-jitted stages | Boundaries between sequentially-called jitted functions; outputs of one stage re-passed as inputs to the next | Fuse the stages under one jit so XLA fuses across the boundary |
| Constant-heavy argument lists (dozens of scalars re-passed every call) | Per-call pytree flatten/validation overhead on values that never change | Physics constants / table handles passed positionally on every stage call | Bake them in via closure/`functools.partial` at wrapper-build time |
| One kernel > 80% of device kernel time | A materialized broadcast — commonly vectorized table interpolation building an `(ncol, nk, table-slice)` intermediate | The op behind the kernel name in the HLO; look for broadcast+select where a gather would do | Rewrite as `jnp.take`/indexed gather |
| Per-column memory ≫ per-column state size (or OOM above some ncol) | A materialized `(ncol, …)` intermediate with a large constant factor | Same broadcast pattern as above; XLA allocation messages give the per-column bytes | Gather/`scan` instead of materializing; this removes the memory-scaling ceiling, not just time |
| H2D/D2H or transpose a large share of the call | Caller-layout conversion and PCIe traffic, not compute | The wrapper/entry conversions (layout transposes, dtype casts) | Device-side transpose (XLA fuses it), `donate_argnums` to reuse buffers, or a caller that passes the native layout |
| Warmup ≫ steady-state call and recompiles appear across calls | Shape-driven recompilation | Varying array shapes or non-static Python values crossing the jit boundary | Pad/bucket shapes; move true constants to `static_argnames`; persistent compile cache |
| Steady-state time flat in ncol while work grows | Fixed host/dispatch floor dominating small problems | Compare the ladder at both ncol points | Same fixes as host-bound dispatch; report the floor and the break-even size |

## Rules for turning a signature into a recommendation

1. **Bounding rule:** a recommendation may never claim a saving larger than
   the ladder layer it lives in. "Fuse the jit boundaries" is bounded by the
   host-dispatch residual; "gather instead of broadcast" by device kernel
   time; "device-side transpose" by the data-movement layer.
2. **Attribution rule:** every recommendation names the code construct
   (file:line). A hotspot without its cause is an observation, not a
   finding.
3. **Cost rule:** state the validation price — any code fix re-triggers the
   full completeness → lint → semantic audit (gate) → runtime → driver → comparison chain — and the cheapest
   measurement that would confirm or kill the hypothesis (often: open the
   `.nsys-rep` timeline and look at the gaps) before the expensive edit.
4. **Rank 3–5, stop.** Ordered by expected payoff per unit of validation
   cost; a longer list dilutes the report.

Historical calibration: the first campaign on a multi-stage orchestrator scheme (2026-08) reached
`production_ready` at iteration 1 while the device computed for 10.5 ms of
a 274 ms call — a 96% device-idle fraction from ~15 sequentially-dispatched
jit stages. The gates were right (nothing pathological) and the headroom
was still 8× end-to-end. That case is why this analysis is mandatory.
