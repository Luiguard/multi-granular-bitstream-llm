#!/bin/bash
cd /home/benjamin/Bilder

echo "========================================================"
echo "🧠 Multi-Granular Bitstream LLM - 8.52B MoE (20-Bit Golden Master)"
echo "========================================================"
echo ""

echo "[1/3] Beende eventuelle alte Hintergrundprozesse..."
pkill -9 -f "autonomous_30day_trainer.py" 2>/dev/null || true
pkill -9 -f "galore_7b_per_layer_trainer.py" 2>/dev/null || true
pkill -9 -f "ingest_chinchilla_optimal.py" 2>/dev/null || true
pkill -9 -f "ingest_7b_massive_expansion.py" 2>/dev/null || true
pkill -9 -f "dashboard/server.py" 2>/dev/null || true
sleep 1

echo "[2/5] Starte 100-Milliarden-Tokens-Pumpe (FineWeb-Edu) im Hintergrund..."
nohup env PYTHONPATH=/home/benjamin/Bilder /home/benjamin/ai-env/bin/python3 scripts/ingest_7b_massive_expansion.py > /home/benjamin/Bilder/ingestion_7b_background.log 2>&1 &
echo "      -> Läuft (Logs: ingestion_7b_background.log)"

echo "[3/5] Starte Kognitiven Takt & Gedankenstrom (Self-Reflection Loop)..."
nohup /home/benjamin/ai-env/bin/python3 pipeline/recurrent_cognitive_loop.py > /home/benjamin/Bilder/recurrent_loop.log 2>&1 &
echo "      -> Läuft (10s Takt · Logs: recurrent_loop.log)"

echo "[4/5] Starte Autonome Selbstrecherche & Wissenslücken-Synthese..."
nohup /home/benjamin/ai-env/bin/python3 pipeline/autonomous_epistemic_learner.py > /home/benjamin/Bilder/epistemic_learner.log 2>&1 &
echo "      -> Läuft (Epistemischer Daemon · Logs: epistemic_learner.log)"

echo "[5/5] Starte Dashboard auf http://localhost:7860 ..."
nohup /home/benjamin/ai-env/bin/python3 dashboard/server.py 7860 > /home/benjamin/Bilder/dashboard_background.log 2>&1 &
echo "      -> Läuft (Logs: dashboard_background.log)"

echo ""
echo "🚀 Starte 8.52B MoE Trainer (20L / 4E à 100.7M / 20-Bit / 3.072 Kontext / Batch 2 / GaLore r=64)..."
echo "--------------------------------------------------------"
sleep 2

# Starte den 8.52B Trainer mit Defragmentierungs-Schutz (3072 Kontext, Batch 2 = 6.144 Tokens/Step, Checkpoint alle 25 Steps)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONPATH=/home/benjamin/Bilder /home/benjamin/ai-env/bin/python3 scripts/galore_7b_per_layer_trainer.py --seq_len 3072 --batch_size 2 --gradient_accumulation_steps 1 --save_interval 25

echo "--------------------------------------------------------"
echo "Trainer wurde beendet. Dieses Fenster schließt sich in 10 Sekunden."
sleep 10
