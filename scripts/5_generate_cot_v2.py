#!/usr/bin/env python3
# ============================================================
# Script 5: Generación de CoT v2 para bit, cipher y algebraic
# Cambios vs v1:
#   - Formato <think></think> correcto para Nemotron
#   - max_tokens=1200 (evita truncamiento en bit)
#   - Verificación y reintento si falta \boxed{}
#   - Guardado incremental cada 100 ejemplos
# ============================================================

import anthropic, json, os, time, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Configuración ────────────────────────────────────────────
API_KEY      = os.environ.get("ANTHROPIC_API_KEY", "TU_API_KEY")
DATA_PATH    = "/workspace/data/training_data.jsonl"
OUTPUT_PATH  = "/workspace/data/cot_v2.jsonl"
CATEGORIES   = ['bit', 'cipher', 'algebraic']
MAX_WORKERS  = 3
MAX_TOKENS   = 1200
MAX_RETRIES  = 2

client = anthropic.Anthropic(api_key=API_KEY)

# ── Prompts por categoría ────────────────────────────────────
SYSTEMS = {
    'bit': """You are solving bit manipulation puzzles. Analyze the input->output examples to find the secret rule.

Your response MUST follow this EXACT format:
<think>
[Your step-by-step reasoning to identify and verify the transformation rule]
</think>
\\boxed{answer}

The answer MUST be an 8-bit binary string inside \\boxed{}. Example: \\boxed{10110011}""",

    'cipher': """You are solving text decryption puzzles. Build a letter mapping table from the examples and decrypt.

Your response MUST follow this EXACT format:
<think>
[Build the letter mapping table and apply it to decrypt the text]
</think>
\\boxed{decrypted text}

The answer MUST be the decrypted text inside \\boxed{}. Example: \\boxed{cat imagines book}""",

    'algebraic': """You are solving symbol transformation puzzles. Find the transformation rules from the examples.

Your response MUST follow this EXACT format:
<think>
[Identify the symbol mappings and apply them to the input]
</think>
\\boxed{result}

The answer MUST be the transformed result inside \\boxed{}. Example: \\boxed{@&}"""
}

# ── Función de generación con reintento ──────────────────────
def gen_cot(prompt, answer, category, attempt=0):
    user_msg = (
        f"{prompt}\n\n"
        f"The correct answer is: {answer}\n"
        f"Generate reasoning that leads to this answer following the EXACT format required."
    )
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": user_msg}],
            system=SYSTEMS[category]
        )
        text = response.content[0].text
        has_think  = '<think>' in text and '</think>' in text
        has_boxed  = '\\boxed{' in text

        if has_think and has_boxed:
            return text
        elif attempt < MAX_RETRIES:
            time.sleep(1)
            return gen_cot(prompt, answer, category, attempt + 1)
        else:
            return None
    except Exception as e:
        if attempt < MAX_RETRIES:
            time.sleep(2)
            return gen_cot(prompt, answer, category, attempt + 1)
        return None

# ── Cargar datos ya generados ────────────────────────────────
done_prompts = set()
if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH) as f:
        for line in f:
            entry = json.loads(line)
            done_prompts.add(entry['messages'][0]['content'])
    print(f"Ya generados: {len(done_prompts)} ejemplos")

# ── Cargar pendientes ────────────────────────────────────────
records = []
with open(DATA_PATH) as f:
    for line in f:
        ex = json.loads(line)
        if ex['category'] in CATEGORIES:
            prompt = ex['messages'][0]['content']
            if prompt not in done_prompts:
                records.append({
                    'prompt':   prompt,
                    'answer':   ex['answer'],
                    'category': ex['category']
                })

print(f"Pendientes: {len(records)} ejemplos")
print(f"Categorías: {pd.Series([r['category'] for r in records]).value_counts().to_dict()}")

# ── Función por fila ─────────────────────────────────────────
def process(record):
    cot = gen_cot(record['prompt'], record['answer'], record['category'])
    if cot:
        return {
            "messages": [
                {"role": "user",      "content": record['prompt']},
                {"role": "assistant", "content": cot}
            ],
            "category": record['category'],
            "answer":   record['answer']
        }
    return None

# ── Generación paralela con guardado incremental ─────────────
results  = []
failed   = 0
start    = time.time()

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(process, r) for r in records]

    for i, future in enumerate(as_completed(futures)):
        result = future.result()
        if result:
            results.append(result)
        else:
            failed += 1

        # Guardar cada 100 ejemplos
        if (i + 1) % 100 == 0:
            with open(OUTPUT_PATH, 'a') as f:
                for entry in results[-min(100, len(results)):]:
                    f.write(json.dumps(entry) + '\n')
            elapsed  = time.time() - start
            rate     = (i + 1) / elapsed
            remaining = (len(records) - (i + 1)) / rate
            print(f"[{i+1}/{len(records)}] guardado — {elapsed/60:.1f}min — ~{remaining/60:.1f}min restantes — fallidos: {failed}")

# Guardar resto
remainder = len(results) % 100
if remainder > 0:
    with open(OUTPUT_PATH, 'a') as f:
        for entry in results[-remainder:]:
            f.write(json.dumps(entry) + '\n')

# ── Resumen ──────────────────────────────────────────────────
with open(OUTPUT_PATH) as f:
    total = sum(1 for _ in f)

print(f"\n✅ Generados: {len(results)}")
print(f"❌ Fallidos:  {failed}")
print(f"✅ Total en disco: {total}")
print(f"⏱️  Tiempo: {(time.time()-start)/60:.1f} min")
