#!/usr/bin/env python3
# ============================================================
# Script 6: Combinar datos determinísticos con CoT v2
# ============================================================

import json, os
from collections import Counter

DET_PATH  = "/workspace/data/cot_deterministic.jsonl"
V2_PATH   = "/workspace/data/cot_v2.jsonl"
OUT_PATH  = "/workspace/data/cot_v2_combined.jsonl"

all_data = []

# Cargar determinísticos
with open(DET_PATH) as f:
    for line in f:
        all_data.append(json.loads(line))
print(f"Determinísticos: {len(all_data)}")

# Cargar CoT v2
v2_count = 0
with open(V2_PATH) as f:
    for line in f:
        all_data.append(json.loads(line))
        v2_count += 1
print(f"CoT v2: {v2_count}")

# Guardar combinado
with open(OUT_PATH, 'w') as f:
    for entry in all_data:
        f.write(json.dumps(entry) + '\n')

cats = Counter(e['category'] for e in all_data)
print(f"\n✅ Total combinado: {len(all_data)}")
print(f"Por categoría: {dict(cats)}")