#!/usr/bin/env python3
# ============================================================
# Script 2: Descarga de datos de entrenamiento desde Kaggle
# Tiempo estimado: ~2 minutos
# ============================================================

import subprocess, os, json

# Instalar kaggle CLI
subprocess.run(["pip", "install", "kaggle", "-q"])

# Configurar credenciales de Kaggle
# Crea el archivo kaggle.json con tus credenciales
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)

KAGGLE_USERNAME = "TU_KAGGLE_USERNAME"  # reemplazar
KAGGLE_KEY      = "TU_KAGGLE_API_KEY"   # reemplazar desde kaggle.com/settings

kaggle_config = {"username": KAGGLE_USERNAME, "key": KAGGLE_KEY}
config_path = os.path.expanduser("~/.kaggle/kaggle.json")
with open(config_path, "w") as f:
    json.dump(kaggle_config, f)
os.chmod(config_path, 0o600)

# Descargar dataset de entrenamiento
os.makedirs("/workspace/data", exist_ok=True)
subprocess.run([
    "kaggle", "datasets", "download",
    "-d", f"{KAGGLE_USERNAME}/nemotron-training-data",
    "-p", "/workspace/data",
    "--unzip"
])

# Verificar descarga
for f in ["training_data.jsonl", "cot_deterministic.jsonl", "cot_api_partial.jsonl"]:
    path = f"/workspace/data/{f}"
    if os.path.exists(path):
        size = os.path.getsize(path) / 1024 / 1024
        count = sum(1 for _ in open(path))
        print(f"✅ {f}: {count} ejemplos ({size:.1f} MB)")
    else:
        print(f"❌ {f}: NO encontrado")
