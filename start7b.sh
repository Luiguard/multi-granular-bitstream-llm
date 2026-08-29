#!/bin/bash
cd /home/benjamin/Bilder

echo "========================================================"
echo "🧠 Multi-Granular Bitstream LLM - 7B Training System"
echo "========================================================"
echo ""

echo "[1/3] Beende eventuelle alte Hintergrundprozesse..."
pkill -f "autonomous_30day_trainer.py"
pkill -f "galore_7b_per_layer_trainer.py"
pkill -f "ingest_chinchilla_optimal.py"
pkill -f "ingest_7b_massive_expansion.py"
pkill -f "dashboard/server.py"
sleep 2

echo "[2/3] Starte 100-Milliarden-Tokens-Pumpe (FineWeb-Edu) im Hintergrund..."
nohup env PYTHONPATH=/home/benjamin/Bilder .venv/bin/python scripts/ingest_7b_massive_expansion.py > /home/benjamin/Bilder/ingestion_7b_background.log 2>&1 &
echo "      -> Läuft (Logs: ingestion_7b_background.log)"

echo "[3/3] Starte Dashboard auf http://localhost:7860 ..."
nohup .venv/bin/python dashboard/server.py 7860 > /home/benjamin/Bilder/dashboard_background.log 2>&1 &
echo "      -> Läuft (Logs: dashboard_background.log)"

echo ""
echo "🚀 Starte 7 Milliarden Parameter Trainer (GaLore & FSDP CPU-Offload)..."
echo "--------------------------------------------------------"
sleep 2

# Starte den 7B Trainer direkt im sichtbaren Terminal mit Defragmentierungs-Schutz
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONPATH=/home/benjamin/Bilder .venv/bin/python scripts/galore_7b_per_layer_trainer.py

echo "--------------------------------------------------------"
echo "Trainer wurde beendet. Dieses Fenster schließt sich in 10 Sekunden."
sleep 10
