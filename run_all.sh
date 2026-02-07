#!/bin/bash
# Esegue tutti i prompt nella cartella prompts/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source .venv/bin/activate

echo "=========================================="
echo "   CogVideoX - Generazione batch"
echo "=========================================="

# Conta i prompt
total=$(ls -1 prompts/*.txt 2>/dev/null | wc -l)
current=0

for prompt_file in prompts/*.txt; do
    # Estrai nome senza percorso e estensione
    name=$(basename "$prompt_file" .txt)
    current=$((current + 1))

    echo ""
    echo "[$current/$total] Generazione: $name"
    echo "=========================================="

    python run_cogvideox.py "$name" "$@"

    echo ""
    echo "✓ Completato: $name"
done

echo ""
echo "=========================================="
echo "   Tutti i video sono stati generati!"
echo "   Output in: outputs/2b/"
echo "=========================================="
