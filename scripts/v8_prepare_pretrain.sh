#!/usr/bin/env bash
# =============================================================================
# v8_prepare_pretrain.sh
#
# STEP 0 (conditional) + STEP 3.3 — Build the v8 warm-start pretrain from the
# v7 checkpoint.
#
# Reads /tmp/v8_vocab_state.txt written by v8_step0_vocab_check.sh:
#   - MISSING empty  : both ॅ and ॲ present -> PURE COPY of v7's model_last.pt
#                      to ckpts/Rasa_Marathi_Emotion_Female_v8/model_extended.pt
#                      (basename MUST be model_extended.pt for finetune_cli).
#   - MISSING nonempty: at least one of ॅ/ॲ absent -> EMBEDDING SURGERY, the
#                      same procedure as the earlier chandrabindu fix:
#                        * append missing char(s) to vocab_extended.txt
#                        * extend BOTH model_state_dict and ema_model_state_dict
#                          text_embed.text_embed.weight by one row per missing
#                          char, via MEAN-init of the existing rows
#                        * drop optimizer/scheduler state
#                        * reset "update" to 0
#
# Verifies (Step 3.3) the output checkpoint's text_embed shape ==
# [vocab_line_count + 1, 512].
#
# NOTE: if surgery is needed, vocab_extended.txt is extended IN PLACE (the
# file shared by future deploys). A timestamped backup is taken first.
#
# Usage (server, after prepare_dataset_v8.sh):
#   bash scripts/v8_prepare_pretrain.sh
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="/root/f5-tts-marathi/f5tts/bin/python3"
CKPT_IN="/root/f5-tts-marathi/f5tts/lib/python3.12/ckpts/Cartesia_Rasa_Combined_v7/model_last.pt"
CKPT_OUT="/root/f5-tts-marathi/f5tts/lib/python3.12/ckpts/Rasa_Marathi_Emotion_Female_v8/model_extended.pt"
VOCAB="/root/f5-tts-marathi/f5tts/data/Rasa_Marathi_Emotion_Female/vocab_extended.txt"
STATE_FILE="/tmp/v8_vocab_state.txt"

for f in "$CKPT_IN" "$VOCAB"; do
    if [[ ! -f "$f" ]]; then
        echo "[ERROR] Required file not found: $f"
        exit 1
    fi
done
if [[ ! -f "$STATE_FILE" ]]; then
    echo "[ERROR] State file missing: $STATE_FILE"
    echo "        Run bash scripts/v8_step0_vocab_check.sh first."
    exit 1
fi

MISSING=$(grep '^MISSING|' "$STATE_FILE" | cut -d'|' -f2 || true)
VOCAB_COUNT=$(grep '^TOKENS|' "$STATE_FILE" | cut -d'|' -f2 || true)

echo "============================================================"
echo " Prepare v8 pretrain (warm start from v7)"
echo "============================================================"
echo "Input  (v7 trained)  : $CKPT_IN"
echo "Output (v8 pretrain) : $CKPT_OUT"
echo "Vocab                : $VOCAB ($VOCAB_COUNT tokens)"
echo "Chars missing        : ${MISSING:-<none>}"

mkdir -p "$(dirname "$CKPT_OUT")"

if [[ -z "$MISSING" ]]; then
    # ------------------------------------------------------------------
    # No surgery: pure copy, then verify shape (Step 3.3)
    # ------------------------------------------------------------------
    echo ""
    echo "[1] No missing char -> pure copy (no embedding surgery)."
    cp -f "$CKPT_IN" "$CKPT_OUT"
    echo "    Copied $CKPT_IN -> $CKPT_OUT"
    SURGERY=0
else
    # ------------------------------------------------------------------
    # Surgery: back up vocab, append missing chars, extend embeddings
    # ------------------------------------------------------------------
    echo ""
    echo "[1] Missing char(s): $MISSING -> embedding surgery required."
    BACKUP_VOCAB="$REPO_ROOT/deploy_backups/vocab_extended_v8_$(date +%Y%m%d_%H%M%S).txt"
    mkdir -p "$REPO_ROOT/deploy_backups"
    cp -p "$VOCAB" "$BACKUP_VOCAB"
    echo "    Vocab backup : $BACKUP_VOCAB"

    export CKPT_IN CKPT_OUT VOCAB MISSING
    "$PYTHON" <<'PY'
import os, torch

ckpt_in, ckpt_out, vocab_path, missing_str = (
    os.environ["CKPT_IN"], os.environ["CKPT_OUT"],
    os.environ["VOCAB"],  os.environ["MISSING"],
)
missing = [c for c in missing_str.split(",") if c]
missing_chars = [chr(int(cp, 16)) for cp in missing]
print("Missing chars to add:", [f"U+{ord(c):04X}" for c in missing_chars])

# --- append missing chars to vocab (preserve order, append at end) --------
existing = []
with open(vocab_path, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if line:
            existing.append(line)
present = set(existing)
added = []
for ch in missing_chars:
    if ch not in present:
        existing.append(ch)
        present.add(ch)
        added.append(ch)
with open(vocab_path, "w", encoding="utf-8", newline="\n") as f:
    for t in existing:
        f.write(t + "\n")
vocab_size = len(existing)
n_add = len(added)
print(f"Vocab now {vocab_size} tokens (+{n_add}: {['U+%04X' % ord(c) for c in added]})")

# --- load checkpoint and extend BOTH state dicts --------------------------
sd = torch.load(ckpt_in, map_location="cpu")
key = "transformer.text_embed.text_embed.weight"
target = vocab_size + 1  # +1 blank/0 token

containers = []
if "model_state_dict" in sd and isinstance(sd["model_state_dict"], dict):
    containers.append(("model_state_dict", sd["model_state_dict"]))
if "ema_model_state_dict" in sd and isinstance(sd["ema_model_state_dict"], dict):
    containers.append(("ema_model_state_dict", sd["ema_model_state_dict"]))
if not containers and isinstance(sd, dict) and key in sd:
    containers.append(("top_level", sd))

if not containers:
    print("ERROR: could not locate text_embed for extension.")
    raise SystemExit(1)

for name, state in containers:
    if key not in state:
        print(f"WARN: {name} has no {key}; skipping.")
        continue
    w = state[key]
    old_num, dim = w.shape[0], w.shape[1]
    if old_num == target:
        print(f"{name}: already {old_num} rows - ok")
        continue
    if old_num > target:
        print(f"ERROR: {name} has {old_num} rows > target {target}; refusing.")
        raise SystemExit(1)
    need = target - old_num
    mean_row = w.mean(dim=0, keepdim=True)  # mean-init for new rows
    new_w = torch.cat([w, mean_row.repeat(need, 1)], dim=0)
    state[key] = new_w
    print(f"{name}: text_embed {old_num} -> {new_w.shape[0]} rows (mean-init +{need})")

# --- drop optimizer/scheduler state, reset update counter ------------------
dropped = []
for k in list(sd.keys()):
    if ("optimizer" in k.lower() or "scheduler" in k.lower()
            or "scaler" in k.lower() or "lr_sched" in k.lower()):
        dropped.append(k)
        del sd[k]
if "update" in sd:
    sd["update"] = 0
    print("reset sd['update'] = 0")
for name, state in containers:
    for k in list(state.keys()):
        if ("optimizer" in k.lower() or "scheduler" in k.lower()
                or "scaler" in k.lower() or "lr_sched" in k.lower()):
            if k not in dropped:
                dropped.append(k)
            del state[k]
print("Dropped optimizer/scheduler keys:", dropped or "(none)")

torch.save(sd, ckpt_out)
print(f"[SAVED] {ckpt_out}")
PY
    SURGERY=1
fi

# ---------------------------------------------------------------------------
# Step 3.3 — verify embedding shape matches [vocab_line_count + 1, 512]
# ---------------------------------------------------------------------------
echo ""
echo "[2] Verifying embedding shape (Step 3.3)..."
export CKPT_OUT VOCAB SURGERY
"$PYTHON" <<'PY'
import os, torch

ckpt_out, vocab_path = os.environ["CKPT_OUT"], os.environ["VOCAB"]
surgery = os.environ.get("SURGERY", "0") == "1"
vocab_size = 0
with open(vocab_path, encoding="utf-8") as f:
    for line in f:
        if line.rstrip("\n"):
            vocab_size += 1
expected = vocab_size + 1

sd = torch.load(ckpt_out, map_location="cpu")
key = "transformer.text_embed.text_embed.weight"
checked = False
for name in ("ema_model_state_dict", "model_state_dict"):
    if name in sd and isinstance(sd[name], dict) and key in sd[name]:
        w = sd[name][key]
        print(f"  {name}.{key}.shape = {list(w.shape)}  (expected [{expected}, 512])")
        assert w.shape[0] == expected, f"embedding rows {w.shape[0]} != expected {expected}"
        checked = True
if not checked and key in sd:
    w = sd[key]
    print(f"  top_level.{key}.shape = {list(w.shape)}  (expected [{expected}, 512])")
    assert w.shape[0] == expected
    checked = True
if not checked:
    print("ERROR: text_embed weight not found for verification.")
    raise SystemExit(1)

print(f"[OK] Pretrain ready: {expected} text embeddings, matches vocab {vocab_size}.")
if surgery:
    print("     Optimizer/scheduler dropped, update reset to 0 (fresh fine-tune).")
PY

echo ""
echo "[INFO] Done. Next: run bash scripts/run_train_v8.sh (after Step 3 pre-flight)."