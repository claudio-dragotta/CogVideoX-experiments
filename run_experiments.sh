#!/bin/bash
# ==========================================================================
#  CogVideoX - Generazione completa esperimenti
#
#  Struttura output:
#    outputs/
#      baseline/                  <- 9 prompt con parametri default
#      exp1_guidance_scale/       <- panda_guitar con GS=1, 6, 12
#      exp2_steps/                <- panda_guitar con steps=10, 25, 50
#      exp3_seed/                 <- panda_guitar con seed=42, 123, 999
#      exp4_frames/               <- panda_guitar con frames=25, 49
# ==========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source .venv/bin/activate

# Prompt usato per tutti gli esperimenti
EXP_PROMPT="panda_guitar"

# Contatore globale
total=0
current=0

# Conta tutti i run
count_runs() {
    local n=0
    # Baseline: tutti i prompt
    n=$((n + $(ls -1 prompts/*.txt 2>/dev/null | wc -l)))
    # Exp1: guidance_scale (3 valori)
    n=$((n + 3))
    # Exp2: steps (3 valori)
    n=$((n + 3))
    # Exp3: seed (3 valori)
    n=$((n + 3))
    # Exp4: frames (2 valori)
    n=$((n + 2))
    echo $n
}

total=$(count_runs)

run_video() {
    local prompt="$1"
    local output="$2"
    shift 2
    # Restanti argomenti sono parametri extra
    current=$((current + 1))

    if [ -f "$output" ]; then
        echo "  [$current/$total] SKIP (esiste gia): $output"
        return
    fi

    echo ""
    echo "  [$current/$total] Generazione: $output"
    python run_cogvideox.py "$prompt" --output "$output" "$@"
    echo "  -> Completato: $output"
}

echo "=========================================="
echo "   CogVideoX - Generazione Esperimenti"
echo "   Totale video da generare: $total"
echo "=========================================="

# ==================================================================
# BASELINE: tutti i 9 prompt con parametri default
# ==================================================================
echo ""
echo "=========================================="
echo "  BASELINE: 9 prompt, parametri default"
echo "  (steps=50, guidance=6, seed=42, frames=49)"
echo "=========================================="

for prompt_file in prompts/*.txt; do
    name=$(basename "$prompt_file" .txt)
    run_video "$name" "outputs/baseline/${name}.mp4"
done

# ==================================================================
# ESPERIMENTO 1: guidance_scale (1, 6, 12)
# ==================================================================
echo ""
echo "=========================================="
echo "  EXP 1: guidance_scale = 1, 6, 12"
echo "  Prompt: $EXP_PROMPT"
echo "  Fissi: steps=50, seed=42, frames=49"
echo "=========================================="

for gs in 1 6 12; do
    run_video "$EXP_PROMPT" \
        "outputs/exp1_guidance_scale/${EXP_PROMPT}_gs${gs}.mp4" \
        --guidance "$gs"
done

# ==================================================================
# ESPERIMENTO 2: num_inference_steps (10, 25, 50)
# ==================================================================
echo ""
echo "=========================================="
echo "  EXP 2: steps = 10, 25, 50"
echo "  Prompt: $EXP_PROMPT"
echo "  Fissi: guidance=6, seed=42, frames=49"
echo "=========================================="

for steps in 10 25 50; do
    run_video "$EXP_PROMPT" \
        "outputs/exp2_steps/${EXP_PROMPT}_steps${steps}.mp4" \
        --steps "$steps"
done

# ==================================================================
# ESPERIMENTO 3: seed (42, 123, 999)
# ==================================================================
echo ""
echo "=========================================="
echo "  EXP 3: seed = 42, 123, 999"
echo "  Prompt: $EXP_PROMPT"
echo "  Fissi: guidance=6, steps=50, frames=49"
echo "=========================================="

for seed in 42 123 999; do
    run_video "$EXP_PROMPT" \
        "outputs/exp3_seed/${EXP_PROMPT}_seed${seed}.mp4" \
        --seed "$seed"
done

# ==================================================================
# ESPERIMENTO 4: num_frames (25, 49)
# ==================================================================
echo ""
echo "=========================================="
echo "  EXP 4: frames = 25, 49"
echo "  Prompt: $EXP_PROMPT"
echo "  Fissi: guidance=6, steps=50, seed=42"
echo "=========================================="

for frames in 25 49; do
    run_video "$EXP_PROMPT" \
        "outputs/exp4_frames/${EXP_PROMPT}_frames${frames}.mp4" \
        --frames "$frames"
done

# ==================================================================
# RIEPILOGO
# ==================================================================
echo ""
echo "=========================================="
echo "   GENERAZIONE COMPLETATA!"
echo "=========================================="
echo ""
echo "Struttura output:"
echo "  outputs/"
echo "    baseline/                  <- 9 prompt, parametri default"
echo "    exp1_guidance_scale/       <- GS=1, 6, 12"
echo "    exp2_steps/                <- steps=10, 25, 50"
echo "    exp3_seed/                 <- seed=42, 123, 999"
echo "    exp4_frames/               <- frames=25, 49"
echo ""
echo "Totale video generati: $current"
echo "=========================================="
