# New Project Checklist — instantiating LAFT for a new Fortran code

How to set up the translation framework for a new Fortran scheme ("XYZ").
The Kessler instance (`../kessler/`) is the worked example for every step.

## 1. Create the project directory

Copy (or, for `ORCHESTRATOR.md`, link) the **generic framework** parts from an
existing instance:

```
laft-xyz/
├── ORCHESTRATOR.md       # SYMLINK -> ../LAFT/ORCHESTRATOR.md, not a copy (see below)
├── config/               # framework_config.py (shared loader) + your
│                         #   project.toml, from the template — see step 2
├── data/src/             # your Fortran source goes here
├── tools/                # clean/copy shell scripts (shared housekeeping)
├── workflow_frontend/    # phases 01-02 (parse/extract) + FRONTEND_WORKFLOW.md
├── workflow_bridge/      # phase03 + BRIDGE_WORKFLOW.md + run_bridge_tests.py
├── workflow_translator/  # phase04 + phase05_01b + prompt_policies/ + TRANSLATE_WORKFLOW.md
├── validation/           # phase05_01/02 (shared validators) + VALIDATION_MANUAL.md
├── workflow_profiler/    # optional stage 3 + PROFILE_WORKFLOW.md + PROFILER_MANUAL.md
├── pbsJobs/              # copy the live jobs; rename the driver pair (step 5)
├── bridge_test/          # OPTIONAL per-project additions (step 3b); the bridge suite itself is shared
└── out/                  # created by the pipeline
```

Do NOT copy: `out/` contents, the reference tree, `translations/`, or any
project policy file (write your own — step 4).

**`ORCHESTRATOR.md` is a symlink, not a copy.** It is a single shared document
that lives in `LAFT/`; every project root just links to it, so edits in `LAFT/`
reach all projects with nothing to sync. Create the link (don't `cp` the file):

```bash
cd laft-xyz
ln -s ../LAFT/ORCHESTRATOR.md ORCHESTRATOR.md   # relative link: laft-xyz must stay a sibling of LAFT/
```

Once the directory exists, `ORCHESTRATOR.md` is the entry point for actually
running the pipeline — you `cd` into the project and run it from there, and the
project is selected entirely by that directory's `config/project.toml` (it is
never named in the orchestrator; see its *One shared orchestrator, run per
project* section). This checklist only covers getting the project set up.

## 2. Write config/project.toml

Copy `config/project.template.toml` to `config/project.toml` and fill in every
TODO. The tools auto-discover it when run from the project root (or pass
`--config`). Start with `[driver].contracts = [1, 2]` ("undecided — certify
both"); pin to the winner only after the project's dual-contract benchmark,
by hand, with a provenance comment (see §6).

## 3. Supply the hand-made inputs

The config makes the pipeline pluggable. What the framework cannot generate is
the **ground truth and the knowledge of the physics** — everything else
(packets, bridges, the whole bridge test suite, lint, audit, runtime smoke
test, comparison metrics, profiler diagnostics) is framework code driven by
metadata and config (`README.md` §Generic vs project-specific, validated end to
end on the Kessler project, 2026-10-09, with zero per-project test code). The hand-made
set is: the four inputs below (always), the physics test in `bridge_test/`
(§3b — the only test a project writes, because sanity ranges and plausible
inputs are physics, not packet data), and, for Stage 3 only, the profiler inputs script
(`[profiler].inputs_script`: how to capture one real bridge call from the
project's driver — see `workflow_profiler/PROFILE_WORKFLOW.md` §Inputs). These four things it cannot generate:

1. **Fortran source** in `data/src/` — one file or several:
   `[source].fortran_files` accepts a list and phase01 writes one merged
   index (phase02_04's topological sort covers cross-file dependencies).
2. **A reference driver tree** (`[reference].dir`) — the original Fortran
   driver (e.g. a column model), its input data, and a captured reference
   output file. This is the ground truth for validation.
3. **A hand-ported JAX driver** (`[driver].script`) — port it from
   `[reference].driver_source` with the same inputs and the same output-file
   format. Rule: the JAX driver mirrors the project's reference driver, never
   another project's. Reusable patterns worth copying from
   `kessler/out/driver/kessler_jax_driver.py`: dynamic module-var metadata
   loading from `out/modules/*.json` and bridge-signature introspection. The driver must
   read the `LAFT_DRIVER_MODE` environment variable (`host` default /
   `device_resident`) and record `"mode"` in its JSON — reference
   implementation `kessler/out/driver/kessler_jax_driver.py` (this repo's paper-1 project);
   Stage 2 greps for the switch before submitting when contract 2 is
   declared.
4. **A comparison script** (`[comparison].script`) — the JAX driver's twin,
   living next to it in `out/driver/`. Comparison is project-specific: its
   input format and pass criteria follow the reference driver's outputs
   (worked example: `kessler/out/driver/kessler_compare_fortran_jax.py` —
   per-variable text files, max-abs-error tolerances). Framework contract,
   enforced by `pbsJobs/jax_gpu_compvalues.sh`: run from the project root;
   exit 0 = PASS, non-zero = FAIL (PBS log authoritative); write a
   human-readable report + a JSON with a top-level PASS/FAIL status next to
   the JAX driver output. The **metrics** are not yours to write: declare the
   two data sets in `[comparison.data]` (layout `per_variable_files` or
   `block_file`) and `validation/compare_metrics.py` computes max abs error,
   MAE, RMSE, rRMSE, corr, bitwise/ULP and more before your script runs
   (`VALIDATION_MANUAL.md` Part 3); your script applies the criterion.

### 3b. Bridge tests — shared; per-project additions optional

Since 2026-10-09 the Stage-1 gate needs **no per-project test file**: the
shared suite in `workflow_bridge/` (`test_generated_bridge_layout.py`,
`test_generated_bridge_device_entry.py`, `test_direction_symmetry.py`)
derives each procedure's contract from the packets and
`run_bridge_tests.py` runs it from any project root (`BRIDGE_WORKFLOW.md`
§Step 2). Add files to `bridge_test/` only for what the packets cannot know:
a `bridge_inputs.py` hook when a kernel is not defined on synthetic data
(the hook convention is documented in
`workflow_bridge/test_generated_bridge_device_entry.py`), a physics/functionality
test, a chain test where procedures share device-resident intermediates, a
closed-form check, a hand-transcribed pin — the 7-item list in
`BRIDGE_WORKFLOW.md` §New project checklist. The gate must report PASS with
zero skips among the layout tests before Stage 2 starts; with
`--require-dependent` (Stage 2, Step 4.5) every translation-dependent test
must also run and pass.

## 4. Optional project-specific knowledge

- **Prompt policy file**: write `workflow_translator/prompt_policies/<XYZ>_rules.md` with
  the scheme's hot loops, lookup-table idioms, and reference constants, then
  list it in
  `[prompts].project_policies`. Leave empty to start — add rules as
  translation failures teach you what the LLM gets wrong.
- **Semantic-audit configuration** (`[semantic_audit]`): enable `lut_indices` /
  `level_conventions` only if the scheme has those structures; set
  `prognostic_vars` (the variables updated as `x = x ± (...)*dt`, with
  `"fortran:python"` aliases for renamed locals) to turn on the update-terms
  check; add one `[[semantic_audit.table_readers]]` block per lookup-table
  reader to turn on the table-read check. The generic checks (calls invoked,
  labelled blocks, zero-forever, direction masks …) need nothing. Absent keys
  are reported as *n/a* in `semantic_audit_report.md`, never silently skipped.
- **Translator (`[translator]`)**: three paths, all documented in
  `workflow_translator/TRANSLATE_WORKFLOW.md` Step 2.
  - `source = "in_context"` (or section absent): the driving agent authors
    the passes itself. The same setting covers *other* agentic CLIs
    (Gemini CLI, Codex) — their output is dropped into `out/jax/` and the
    exact model goes into `llm_label` (the LLM-version header on every file
    is mandatory). **Isolation rule for independent-translation claims:**
    an agentic run can read every archive under `translations/`; run it in
    a checkout with `translations/` absent, and require it to save the
    `{proc}_passN.py` intermediates so the pass trajectory is auditable.
  - Locally-hosted vLLM model on PBS (the paper's Qwen2.5-Coder-32B, or
    another Qwen model): set `source`, `module`, `llm_label`,
    `pbs_translate_job`, `pbs_translate_job_name`; copy the
    external-translator folder (`workflow_translator/<model>/` — module doc
    `<model>.md` + batch driver) and its PBS job (`pbs_qwen25_translate.sh`
    fixes the paper's sampling: T = 0.1, unseeded, no reasoning mode) and
    every request's budget is logged to
    `out/reports/analysis/generation_log.jsonl`. Worked instance:
    `kessler/config/project.toml`.

## 5. PBS jobs

The driver pair (`jax_driver.sh` / `jax_driver_gpu.sh`) is generic: it reads
`[driver].script`, `output_file`, and `summary_json` from `config/project.toml`,
so copy it unchanged and keep `[hpc].driver_job`/`driver_job_name` at their
template defaults (only override for custom driver jobs). `jax_gpu_runtimevalid.sh`
and `jax_gpu_compvalues.sh` are generic — copy unchanged. Update `#PBS -A/-q`,
modules, and conda env for your cluster in the scripts' headers (they are set
there directly, not in config). **Always submit from the project root** — the
scripts cd to `$PBS_O_WORKDIR` and abort if `config/project.toml` is missing.

## 6. Run the pipeline

Enter through **`ORCHESTRATOR.md`** (the symlink in the project root) — it
owns the stage order, the entry gates and the mandatory approval pause
between stages, and it is the only supported way to start a run. For
reference, the commands it drives are:

```bash
cd laft-xyz
# Stage 0 — frontend
python workflow_frontend/phase01_01_ts_parse.py          # parse      (needs tree-sitter env `ts`)
python workflow_frontend/phase02_01_procs_JSON.py        # packets
python workflow_frontend/phase02_02_deps_JSON.py         # deps       (needs tree-sitter env `ts`)
python workflow_frontend/phase02_03_merge_packets_JSON.py
python workflow_frontend/phase02_04_module_dependencies.py
# Stage 1 — bridge + gate
python workflow_bridge/phase03_make_bridge.py
python workflow_bridge/run_bridge_tests.py --require-dependent
# Stage 2 — translate (prompts, then TRANSLATE_WORKFLOW.md)
python workflow_translator/phase04_01_make_prompts.py
python workflow_translator/phase04_02_make_wrappers.py     # wrapper skeletons -> out/wrappers/ (reference only)
```

Stage 2 then follows `workflow_translator/TRANSLATE_WORKFLOW.md`: four
passes per prompt chain, completeness check after each procedure
(`phase04_04` — a scaffold stops the run for a human decision), lint
(`phase05_01`), semantic audit (`phase05_01b`, gated by `audit_gate.py` —
runtime validation is refused until it is clean), runtime validation (PBS),
bridge suite with the dependent tests, driver + comparison (PBS, once per
declared contract mode).

**Then decide the contract.** With `[driver].contracts = [1, 2]` both entries are certified by the
driver + comparison step; benchmark them on the project's own grid; pin the
winner by hand with a provenance comment in `config/project.toml`. A pinned
project runs only its declared contract from then on (`kessler/` is pinned to
`[1]`, host arrays in/out). Stage 3 (optional) is
`workflow_profiler/PROFILE_WORKFLOW.md`. After a run: archive with
`tools/copy_AI_results.sh` → `translations/<llm>/`, then
`tools/clean_AI_results.sh` before the next LLM.

## 7. Things to know

- **Never rerun phases 01–02 after curating packets**: hand-set packet edits
  (e.g. `jax_required` flags) would be regenerated away. Regenerate only in a
  scratch copy (git worktree) if you need to compare. (`phase04_02` is safe to
  rerun: its skeletons go to `out/wrappers/`, never `out/jax/`.)
- The pipeline needs two conda envs on Derecho: `ts` (tree-sitter, phases
  01/02.2) and `jax-validate` (everything else, including the bridge suite;
  has `tomli` for the config). Locally-served translators use their own env
  (see the model's `<model>.md`).
- Edit prompt policies in `workflow_translator/prompt_policies/`, never the generated
  `out/prompts/*` files; regenerate with `phase04_01`.
- Layout convention: the bridge converts `(ncol,nz) <-> (nz,ncol)` on the
  device, inside the jitted region; translated code is row-major only, no
  `swapaxes` (`workflow_translator/prompt_policies/layout_policy.md`).
- Never point a new translation at another project's driver or at an
  earlier archive: the driver mirrors *this* project's reference driver,
  and archives stay out of the agent's workspace during authoring.
