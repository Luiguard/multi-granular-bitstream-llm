#!/bin/bash
cd /home/benjamin/Bilder

echo "========================================================"
echo "🧠 Multi-Granular Bitstream LLM - 1.6B Training System"
echo "========================================================"
echo ""

echo "[1/3] Beende eventuelle alte Hintergrundprozesse..."
pkill -f "autonomous_30day_trainer.py"
pkill -f "ingest_chinchilla_optimal.py"
pkill -f "dashboard/server.py"
sleep 2

echo "[2/3] Starte Daten-Ingestion (Premium Datasets) im Hintergrund..."
nohup .venv/bin/python scripts/ingest_chinchilla_optimal.py > /home/benjamin/Bilder/ingestion_background.log 2>&1 &
echo "      -> Läuft (Logs: ingestion_background.log)"

echo "[3/3] Starte Dashboard auf http://localhost:7860 ..."
nohup .venv/bin/python dashboard/server.py 7860 > /home/benjamin/Bilder/dashboard_background.log 2>&1 &
echo "      -> Läuft (Logs: dashboard_background.log)"

echo ""
echo "🚀 Starte 1.6 Milliarden Parameter Trainer..."
echo "--------------------------------------------------------"
sleep 2

# Starte den Trainer direkt im sichtbaren Terminal
PYTHONUNBUFFERED=1 PYTHONPATH=/home/benjamin/Bilder .venv/bin/python scripts/autonomous_30day_trainer.py --days 30.0 --use_moe --batch_size 2 --gradient_accumulation_steps 64

echo "--------------------------------------------------------"
echo "Trainer wurde beendet. Dieses Fenster schließt sich in 10 Sekunden."
sleep 10
