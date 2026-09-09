#!/bin/bash
set -euo pipefail

# Determine project root (parent of this script's directory)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"

# Valid target options: [llm].copy_targets in config/project.toml
# (single-line TOML array, e.g.  copy_targets = ["claude", "gpt", ...])
mapfile -t VALID_TARGETS < <(
  sed -n 's/^copy_targets *= *\[\(.*\)\]/\1/p' "$PROJECT_ROOT/config/project.toml" \
    | tr ',' '\n' | tr -d ' "'
)
if [ "${#VALID_TARGETS[@]}" -eq 0 ]; then
    echo "ERROR: could not read [llm].copy_targets from $PROJECT_ROOT/config/project.toml"
    exit 1
fi

FROM_DIR="$PROJECT_ROOT"
# Safety checks
if [ ! -d "$FROM_DIR" ]; then
    echo "ERROR: '$FROM_DIR' not found. Aborting."
    exit 1
fi

# Read target parameter (default = claude)
TARGET="${1:-}"

if [ -z "$TARGET" ]; then
    echo "Usage: $0 <target>"
    echo "Valid targets: ${VALID_TARGETS[*]}"
    echo "Default: claude"
    echo ""
    echo "Select target:"
    select TARGET in "${VALID_TARGETS[@]}"; do
        if [ -n "$TARGET" ]; then
            break
        fi
        echo "Invalid selection. Try again."
    done
fi

# Validate target
VALID=false
for t in "${VALID_TARGETS[@]}"; do
    if [ "$TARGET" = "$t" ]; then
        VALID=true
        break
    fi
done
if [ "$VALID" = false ]; then
    echo "ERROR: Invalid target '$TARGET'. Valid targets: ${VALID_TARGETS[*]}"
    exit 1
fi

# Safety check for target directory
TO_DIR="$PROJECT_ROOT/translations/$TARGET"
if [ ! -d "$TO_DIR" ]; then
    echo "ERROR: Target directory '$TO_DIR' not found. Aborting."
    exit 1
fi

TO_DIR="$PROJECT_ROOT/translations/$TARGET"

echo "Project root detected as:"
echo "  $PROJECT_ROOT"
echo ""
echo "The following directories will be copied:"
echo "  $FROM_DIR/out/jax/"
echo "  $FROM_DIR/out/driver/"
echo "  $FROM_DIR/out/validation/"
echo "  $FROM_DIR/out/lint/"
echo "  $FROM_DIR/out/prompts/"
echo "  $FROM_DIR/out/issues/   (kept as its own issues/ folder, mirroring out/)"
echo "  $FROM_DIR/out/reports/  (consolidated workflow reports: bridge/translation/profile)"
echo "  $FROM_DIR/out/bridge/"
echo "  $FROM_DIR/out/profiled/  (if present — PROFILE_WORKFLOW.md artifacts)"
echo ""
echo "Destination:"
echo "  $TO_DIR"
echo ""

read -r -p "Type YES to confirm copying: " CONFIRM

if [ "$CONFIRM" != "YES" ]; then
    echo "Aborted."
    exit 0
fi

echo "Creating destination directories..."
mkdir -p \
    "$TO_DIR/prompts" \
    "$TO_DIR/jax" \
    "$TO_DIR/driver/results" \
    "$TO_DIR/validation/results" \
    "$TO_DIR/lint/results" \
    "$TO_DIR/issues" \
    "$TO_DIR/reports"

echo "Syncing generated files..."

rsync -av "$FROM_DIR/out/prompts/"    "$TO_DIR/prompts/"
rsync -av "$FROM_DIR/out/jax/"        "$TO_DIR/jax/"
rsync -av "$FROM_DIR/out/driver/"     "$TO_DIR/driver/results/"
rsync -av "$FROM_DIR/out/validation/" "$TO_DIR/validation/results/"
rsync -av "$FROM_DIR/out/lint/"       "$TO_DIR/lint/results/"
# out/issues/ keeps its own identity in the archive (mirrors out/): the fix
# log, the semantic-audit JSONs, and the waivers are working defect records,
# not workflow reports, so they do not get mixed into reports/.
rsync -av "$FROM_DIR/out/issues/"     "$TO_DIR/issues/"
# NOTE: out/bridge/ is NOT snapshotted anywhere — clean_AI_results.sh never
# deletes it, so the live copy in out/bridge/ survives every LLM switch. If it
# is ever lost, regenerate with phase03 + the bridge test gate.

# out/reports/ (consolidated workflow reports) maps 1:1 onto $TO_DIR/reports/,
# which now holds ONLY the workflow reports (bridge/ translation/ profile/).
if [ -d "$FROM_DIR/out/reports" ]; then
    rsync -av "$FROM_DIR/out/reports/" "$TO_DIR/reports/"
fi

# out/profiled/ (PROFILE_WORKFLOW.md) is optional — not every archived run
# went through the profiler workflow.
if [ -d "$FROM_DIR/out/profiled" ]; then
    mkdir -p "$TO_DIR/profiled"
    rsync -av --exclude='*.pkl' --exclude='capture_tmp/' --exclude='iteration_*/hlo_dump/' --exclude='iteration_*/jax_trace*/' \
        "$FROM_DIR/out/profiled/" "$TO_DIR/profiled/"
fi
echo "Sync complete."