#!/bin/bash
set -euo pipefail

# Clears the current translator's out/ artifacts to prepare for a fresh
# translation run. Mirrors copy_AI_results.sh's directory mapping, so before
# deleting anything it verifies every file that would be deleted already has
# a byte-identical copy in translations/<llm>/ — not just that the archive
# directory exists, but that its content actually matches what's in out/
# right now. Run tools/copy_AI_results.sh <llm> first if this refuses to
# proceed.
#
# Preserved, never deleted:
#   out/bridge/  (ENTIRE directory)               — the Fortran↔JAX bridge is
#     generated ONCE by phase03 and is LLM-INDEPENDENT: one correct bridge is
#     reused by every translator. If bridge generation ever had a defect and
#     the bridge was hand-fixed during the first LLM's run, that fix MUST
#     survive into the next LLM's run — so clean keeps the whole directory.
#     (out/bridge/ is the ONLY copy; nothing snapshots it elsewhere.) To get
#     a fresh bridge, re-run phase03 deliberately; clean never removes it.
#   out/reports/bridge/                           — the bridge's phase03
#     analysis reports + test-gate verdict, tied to the (preserved, shared)
#     bridge code, so they persist across LLM switches too. copy_AI_results.sh
#     still snapshots them per-LLM (translations/<llm>/reports/bridge/), just
#     in case. The per-LLM reports (out/reports/translation/ and
#     out/reports/profile/) ARE cleaned.
#   out/issues/semantic_audit_waivers.json        — the semantic audit's
#     waivers are judgments about the FORTRAN source (a construct the audit
#     flags but that is correct as translated), not about any one LLM's
#     output, so they are LLM-independent the way out/bridge/ is. Deleting
#     them would silently re-raise every waived finding for the next
#     translator and cost the same re-adjudication again. The rest of
#     out/issues/ (fix_log.md, audit results) IS cleaned.
#   out/jax/__init__.py                           — package marker; no phase
#     tool regenerates it, so deleting it would silently break every
#     `out.jax.<proc>` import for the NEXT translator's run.
#   out/profiled/*.pkl                            — the profiler's
#     captured driver state is derived from the driver's real input data,
#     not from any translation, so it's identical and reusable across
#     every LLM; clearing it would just force an unnecessary re-capture.
#   out/profiled/.gitignore                       — project-level config,
#     not a translation artifact.
#   out/profiled/<basename of [profiler].inputs_script>  — the project's
#     hand-authored capture_state/tile_state file (e.g.
#     kessler_profiler_inputs.py); driver-specific, not LLM-specific, and
#     nothing regenerates it.
#   out/driver/<basenames of [driver].script and [comparison].script> —
#     the hand-authored per-project driver and comparison scripts; they
#     live next to generated results but are LLM-independent, and the
#     driver/comparison PBS jobs fail without them.
#
# Untouched entirely (Fortran-source-derived, identical regardless of
# translator, not part of what copy_AI_results.sh archives):
#   out/modules/ out/packets/ out/procedures/ out/programs/ out/wrappers/
#   out/phase1_index.json out/module_dependencies.json
#   out/procedure_call_graph.svg

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"
OUT_DIR="$PROJECT_ROOT/out"

if [ ! -d "$OUT_DIR" ]; then
    echo "ERROR: '$OUT_DIR' not found. Aborting."
    exit 1
fi

# out/ is always the ONE currently-active translator's workspace. The
# archive directory to verify against is $1 when given (same convention as
# copy_AI_results.sh <target> — needed e.g. when [translator].source is
# "in_context", a mode rather than an LLM name); otherwise it defaults to
# [translator].source in config/project.toml.
LLM="${1:-}"
if [ -z "$LLM" ]; then
    LLM="$(sed -n 's/^source *= *"\([^"]*\)".*/\1/p' "$PROJECT_ROOT/config/project.toml" | head -1)"
fi
if [ -z "$LLM" ]; then
    echo "ERROR: no target given and could not read [translator].source from $PROJECT_ROOT/config/project.toml"
    exit 1
fi
LLM_DIR="$PROJECT_ROOT/translations/$LLM"
# NOTE: out/bridge/ is intentionally NOT verified or cleaned here — it is
# preserved across LLM switches (see the header) and is the only copy.

# [profiler].inputs_script (if set) is hand-authored per-project
# infrastructure, not a translation artifact — never delete it.
INPUTS_SCRIPT_BASENAME=""
INPUTS_SCRIPT_PATH="$(sed -n 's/^inputs_script *= *"\([^"]*\)".*/\1/p' "$PROJECT_ROOT/config/project.toml" | head -1)"
[ -n "$INPUTS_SCRIPT_PATH" ] && INPUTS_SCRIPT_BASENAME="$(basename "$INPUTS_SCRIPT_PATH")"

# [driver].script and [comparison].script are hand-authored per-project
# infrastructure too — they live in out/driver/ next to generated results,
# but they are LLM-independent and nothing regenerates them (the driver PBS
# job fails without them). Collect every `script = "..."` basename in the
# config to preserve inside out/driver/.
mapfile -t DRIVER_SCRIPT_BASENAMES < <(
  sed -n 's/^script *= *"\([^"]*\)".*/\1/p' "$PROJECT_ROOT/config/project.toml" \
    | xargs -r -n1 basename
)

echo "Project root detected as:"
echo "  $PROJECT_ROOT"
echo "Archive target: $LLM"
echo ""

# --- Safety check: everything about to be deleted must already be archived,
#     with matching content (not just present) -----------------------------
MISSING=0

check_archived() {
    # $1 = out/ source dir   $2 = translations/... destination dir
    # $3... = extra diff -x exclude patterns (glob, no path)
    local src="$1" dst="$2"
    shift 2
    local excludes=(-x '__init__.py' -x '__pycache__')
    for e in "$@"; do
        excludes+=(-x "$e")
    done

    [ -d "$src" ] || return 0   # nothing here to clean, nothing to verify

    if [ ! -d "$dst" ]; then
        echo "  NOT ARCHIVED: $dst does not exist (source: $src)"
        MISSING=1
        return
    fi

    local diffout
    diffout="$(diff -rq "${excludes[@]}" "$src" "$dst" 2>&1 \
        | grep -v "^Only in $dst" || true)"
    if [ -n "$diffout" ]; then
        echo "  NOT FULLY ARCHIVED: $src vs $dst"
        echo "$diffout" | sed 's/^/    /'
        MISSING=1
    fi
}

echo "Verifying everything to be cleaned is archived (matching content) in"
echo "translations/$LLM/ ..."
check_archived "$OUT_DIR/prompts"    "$LLM_DIR/prompts"
check_archived "$OUT_DIR/jax"        "$LLM_DIR/jax"
check_archived "$OUT_DIR/driver"     "$LLM_DIR/driver/results"
check_archived "$OUT_DIR/validation" "$LLM_DIR/validation/results"
check_archived "$OUT_DIR/lint"       "$LLM_DIR/lint/results"
# out/issues/ archives to translations/<llm>/issues/, keeping its own name
# (it holds defect records, not workflow reports). semantic_audit_waivers.json
# is preserved (LLM-independent), so it is excluded from the pre-delete check.
check_archived "$OUT_DIR/issues"     "$LLM_DIR/issues"    'semantic_audit_waivers.json'
# out/reports/bridge/ is preserved (tied to the shared bridge), so it is
# excluded from the pre-delete check — only translation/ and profile/ get
# cleaned.
check_archived "$OUT_DIR/reports"    "$LLM_DIR/reports"   'bridge'
# out/bridge/ is preserved, not cleaned — no pre-delete verification needed.
check_archived "$OUT_DIR/profiled"   "$LLM_DIR/profiled" \
    '*.pkl' '.gitignore' 'capture_tmp' 'hlo_dump' 'jax_trace*'

if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "ERROR: out/ is not fully archived to translations/$LLM/."
    echo "Run 'tools/copy_AI_results.sh $LLM' first, then re-run this script."
    exit 1
fi
echo "  OK — everything is archived with matching content."
echo ""

echo "The following will be cleaned (contents; the ENTIRE out/bridge/ and"
echo "out/reports/bridge/, __init__.py, out/profiled/*.pkl, .gitignore, the"
echo "[profiler].inputs_script file, and the hand-authored [driver].script /"
echo "[comparison].script files in out/driver/ are preserved):"
echo "  $OUT_DIR/prompts/"
echo "  $OUT_DIR/jax/"
echo "  $OUT_DIR/driver/"
echo "  $OUT_DIR/validation/"
echo "  $OUT_DIR/lint/"
echo "  $OUT_DIR/issues/  (except semantic_audit_waivers.json)"
echo "  $OUT_DIR/reports/  (except reports/bridge/)"
echo "  $OUT_DIR/profiled/"
echo ""

read -r -p "Type YES to confirm deletion: " CONFIRM

if [ "$CONFIRM" != "YES" ]; then
    echo "Aborted."
    exit 0
fi

echo "Deleting generated files..."

rm -rf "$OUT_DIR/prompts/"*
find "$OUT_DIR/jax"    -mindepth 1 ! -name '__init__.py' -exec rm -rf {} +

# out/driver/ mixes generated results with the hand-authored per-project
# driver/comparison scripts — delete only the generated results.
DRIVER_KEEP=()
for b in ${DRIVER_SCRIPT_BASENAMES[@]+"${DRIVER_SCRIPT_BASENAMES[@]}"}; do
    DRIVER_KEEP+=(! -name "$b")
done
find "$OUT_DIR/driver" -mindepth 1 ${DRIVER_KEEP[@]+"${DRIVER_KEEP[@]}"} -exec rm -rf {} +

rm -rf "$OUT_DIR/validation/"*
rm -rf "$OUT_DIR/lint/"*
# semantic_audit_waivers.json is a judgment about the Fortran, shared across
# LLMs — keep it; clean only this run's issue artifacts.
if [ -d "$OUT_DIR/issues" ]; then
    find "$OUT_DIR/issues" -mindepth 1 ! -name 'semantic_audit_waivers.json' -exec rm -rf {} +
fi
# out/reports/bridge/ is tied to the preserved, shared bridge — keep it;
# clean only the per-LLM reports (translation/, profile/).
if [ -d "$OUT_DIR/reports" ]; then
    find "$OUT_DIR/reports" -mindepth 1 -maxdepth 1 ! -name 'bridge' -exec rm -rf {} +
fi
# out/bridge/ is deliberately NOT cleaned — the bridge is generated once and
# shared across all LLMs; a hand-fix from one run must persist to the next.

if [ -d "$OUT_DIR/profiled" ]; then
    PROFILED_KEEP=(! -name '*.pkl' ! -name '.gitignore')
    if [ -n "$INPUTS_SCRIPT_BASENAME" ]; then
        PROFILED_KEEP+=(! -name "$INPUTS_SCRIPT_BASENAME")
    fi
    find "$OUT_DIR/profiled" -mindepth 1 "${PROFILED_KEEP[@]}" -exec rm -rf {} +
fi

# Self-heal: recreate the package markers if they were ever missing (no
# phase tool regenerates these).
[ -f "$OUT_DIR/jax/__init__.py" ]    || echo "# Auto-generated package marker" > "$OUT_DIR/jax/__init__.py"
[ -f "$OUT_DIR/bridge/__init__.py" ] || echo "# Auto-generated package marker" > "$OUT_DIR/bridge/__init__.py"

echo "Clean complete."
