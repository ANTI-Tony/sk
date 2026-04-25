#!/usr/bin/env bash
#
# One-shot: set up env, download GoS data, install harbor, run a 4-rollout smoke
# test (1 query x 4 bundles), then estimate the cost of the full 80-run sanity
# check.
#
# Usage:
#   bash setup_and_smoke.sh <ANTHROPIC_API_KEY>
#
# Re-runnable: skips downloads / installs that already exist.
#

set -euo pipefail

# --- args ---
if [ $# -lt 1 ]; then
    echo "Usage: bash setup_and_smoke.sh <ANTHROPIC_API_KEY>" >&2
    exit 1
fi
API_KEY="$1"
if ! [[ "$API_KEY" =~ ^sk-ant- ]]; then
    echo "ERROR: API key must start with 'sk-ant-'." >&2
    exit 1
fi

# --- paths ---
SK_EX_ROOT="$HOME/Tony_wjb/sk_ex"
GOS_DIR="$SK_EX_ROOT/graph-of-skills"
SANITY_DIR="$SK_EX_ROOT/gos-sanity"

if [ ! -d "$GOS_DIR" ]; then
    echo "ERROR: GoS clone not found at $GOS_DIR" >&2
    exit 1
fi
if [ ! -d "$SANITY_DIR" ]; then
    echo "ERROR: gos-sanity clone not found at $SANITY_DIR" >&2
    exit 1
fi

step() { printf "\n========== STEP %s: %s ==========\n" "$1" "$2"; }

# ============================================================
step 1 "Set ANTHROPIC_API_KEY in this shell + persist to ~/.bashrc"
# ============================================================
export ANTHROPIC_API_KEY="$API_KEY"
export ANTHROPIC_AUTH_TOKEN="$API_KEY"

if ! grep -q 'ANTHROPIC_API_KEY=' ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc <<EOF
export ANTHROPIC_API_KEY='$API_KEY'
export ANTHROPIC_AUTH_TOKEN='$API_KEY'
EOF
    echo "Added ANTHROPIC_API_KEY to ~/.bashrc"
else
    echo "~/.bashrc already has ANTHROPIC_API_KEY (not overwriting)"
fi

# Verify key works
echo ">>> Verifying API key..."
RESP=$(curl -s -w "\n%{http_code}" https://api.anthropic.com/v1/messages \
    -H "x-api-key: $ANTHROPIC_API_KEY" \
    -H "anthropic-version: 2023-06-01" \
    -H "content-type: application/json" \
    -d '{"model":"claude-sonnet-4-5","max_tokens":20,"messages":[{"role":"user","content":"say pong"}]}')
HTTP_CODE=$(echo "$RESP" | tail -1)
BODY=$(echo "$RESP" | sed '$d')
echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
if [ "$HTTP_CODE" != "200" ]; then
    echo "ERROR: API returned HTTP $HTTP_CODE" >&2
    exit 1
fi
echo ">>> API key OK"

# ============================================================
step 2 "Download GoS minimal data (skills_200 + workspace + tasks)"
# ============================================================
cd "$GOS_DIR"
mkdir -p data/skillsets data/gos_workspace evaluation/skillsbench

if [ -d data/skillsets/skills_200 ] && [ "$(ls data/skillsets/skills_200 | wc -l)" -gt 100 ]; then
    echo "skills_200 already present ($(ls data/skillsets/skills_200 | wc -l) skills), skipping"
else
    echo ">>> Downloading skills_200 from hf-mirror..."
    curl -fL --retry 3 --max-time 600 -o /tmp/skills_200.tar.gz \
        https://hf-mirror.com/datasets/DLPenn/graph-of-skills-data/resolve/main/skills_200.tar.gz
    [ -s /tmp/skills_200.tar.gz ] || { echo "ERROR: skills_200 download empty" >&2; exit 1; }
    ls -lh /tmp/skills_200.tar.gz
    tar xzf /tmp/skills_200.tar.gz -C data/skillsets/
    echo "skills_200 count: $(ls data/skillsets/skills_200/ | wc -l)"
fi

if [ -d data/gos_workspace/skills_200_v1 ] && [ "$(ls data/gos_workspace/skills_200_v1 | wc -l)" -gt 0 ]; then
    echo "gos_workspace/skills_200_v1 already present, skipping"
else
    echo ">>> Downloading gos_workspace_skills_200_v1 from hf-mirror..."
    curl -fL --retry 3 --max-time 600 -o /tmp/ws.tar.gz \
        https://hf-mirror.com/datasets/DLPenn/graph-of-skills-data/resolve/main/gos_workspace_skills_200_v1.tar.gz
    [ -s /tmp/ws.tar.gz ] || { echo "ERROR: workspace download empty" >&2; exit 1; }
    ls -lh /tmp/ws.tar.gz
    tar xzf /tmp/ws.tar.gz -C data/gos_workspace/
fi

if [ -d evaluation/skillsbench/tasks ] && [ "$(ls evaluation/skillsbench/tasks | wc -l)" -gt 50 ]; then
    echo "SkillsBench tasks already present ($(ls evaluation/skillsbench/tasks | wc -l) tasks), skipping"
else
    echo ">>> Downloading SkillsBench tasks (~580MB) via gh-proxy..."
    curl -fL --retry 3 --max-time 1800 -o /tmp/sb.tar.gz \
        https://gh-proxy.com/https://github.com/benchflow-ai/skillsbench/archive/refs/heads/main.tar.gz
    [ -s /tmp/sb.tar.gz ] || { echo "ERROR: skillsbench download empty" >&2; exit 1; }
    ls -lh /tmp/sb.tar.gz
    tar xzf /tmp/sb.tar.gz -C /tmp/
    rm -f evaluation/skillsbench/tasks
    ln -sfn /tmp/skillsbench-main/tasks evaluation/skillsbench/tasks
    echo "task count: $(ls evaluation/skillsbench/tasks/ | wc -l)"
fi

# ============================================================
step 3 "Install GoS Python deps via uv"
# ============================================================
cd "$GOS_DIR"

if ! command -v uv >/dev/null 2>&1; then
    echo ">>> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

uv sync

# ============================================================
step 4 "Install Harbor"
# ============================================================
if ! command -v harbor >/dev/null 2>&1; then
    echo ">>> Installing harbor..."
    uv tool install harbor
    export PATH="$HOME/.local/bin:$PATH"
fi
harbor --version || { echo "ERROR: harbor install failed" >&2; exit 1; }

# ============================================================
step 5 "Set up gos-sanity venv + install"
# ============================================================
cd "$SANITY_DIR"
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -e .
# Make GoS importable from our venv (installs its deps into .venv)
pip install --quiet -e "$GOS_DIR" || {
    echo "WARN: pip install -e graph-of-skills failed; gos_interface will fall back to PYTHONPATH"
    export PYTHONPATH="$GOS_DIR:${PYTHONPATH:-}"
}

# ============================================================
step 6 "Select queries + truncate to 1 for smoke test"
# ============================================================
python scripts/select_queries.py

python3 - <<'PY'
import json, pathlib
p = pathlib.Path('data/queries.json')
qs = json.loads(p.read_text())
p.write_text(json.dumps(qs[:1], indent=2, ensure_ascii=False))
print(f"Smoke test query: {qs[0]['id']}")
print(f"Prompt preview: {qs[0]['query'][:200]}")
PY

# ============================================================
step 7 "Run smoke test (4 rollouts)"
# ============================================================
mkdir -p results
# Wipe stale runs.jsonl so cost extrapolation only sees this smoke run
> results/runs.jsonl
python scripts/run_experiment.py 2>&1 | tee /tmp/smoke.log

# ============================================================
step 8 "Token / cost extrapolation for full 80-run experiment"
# ============================================================
python3 - <<'PY'
import json, pathlib
runs = [json.loads(l) for l in pathlib.Path('results/runs.jsonl').read_text().splitlines() if l.strip()]
print(f"\nRuns completed: {len(runs)}")
for r in runs:
    print(f"  {r['bundle_type']:20s} reward={r.get('reward')} tokens={r.get('token_count')} time={r['execution_time']:.1f}s err={r.get('error_type')}")

toks = [r['token_count'] for r in runs if r.get('token_count')]
if toks:
    avg = sum(toks) / len(toks)
    cost_per_run = (avg * 0.95 * 3 + avg * 0.05 * 15) / 1_000_000
    print(f"\nAvg total tokens / run: {avg:.0f}")
    print(f"Estimated cost / run:    ${cost_per_run:.2f}")
    print(f"Extrapolate 80 runs:     ${cost_per_run * 80:.0f}")
    print()
    if cost_per_run * 80 > 480:
        print("STOP: extrapolated cost > $480. Investigate before running full.")
    elif cost_per_run * 80 > 240:
        print("CAUTION: cost in $240-480 range. Tighten timeout or accept the cost.")
    else:
        print("OK: extrapolated cost <= $240. Safe to top up balance and run full.")
else:
    print("WARN: no token_count in any run; check error_type and harbor logs.")
PY

echo ""
echo "Smoke test done. Results in results/runs.jsonl. Plot will be written"
echo "after the full experiment by: python scripts/analyze.py"
