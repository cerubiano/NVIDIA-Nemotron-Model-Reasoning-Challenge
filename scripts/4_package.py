#!/usr/bin/env python3
# ============================================================
# Script 4: Empaquetar adapter para submission en Kaggle
# ============================================================

import os, zipfile, shutil

ADAPTER_DIR = "/workspace/adapter"
ZIP_PATH    = "/workspace/submission.zip"

# Copiar trainer_state.json si existe
trainer_state = "/workspace/trainer_state.json"
if os.path.exists(trainer_state):
    shutil.copy(trainer_state, ADAPTER_DIR)
    print("✅ trainer_state.json incluido")

# Verificar archivos requeridos
required = ["adapter_config.json", "adapter_model.safetensors"]
print("Verificando archivos del adapter...")
for f in required:
    path = os.path.join(ADAPTER_DIR, f)
    if os.path.exists(path):
        size = os.path.getsize(path) / 1024 / 1024
        print(f"  ✅ {f} ({size:.1f} MB)")
    else:
        raise FileNotFoundError(f"❌ CRÍTICO: {f} no encontrado. El submission fallará.")

# Crear zip
print(f"\nCreando {ZIP_PATH}...")
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in os.listdir(ADAPTER_DIR):
        fpath = os.path.join(ADAPTER_DIR, fname)
        zf.write(fpath, fname)
        print(f"  Agregado: {fname}")

size = os.path.getsize(ZIP_PATH) / 1024 / 1024
print(f"\n✅ submission.zip creado ({size:.1f} MB)")
print("Listo para subir a Kaggle.")