#!/usr/bin/env python3
# ============================================================
# Script 3: Entrenamiento SFT con LoRA
# Modelo: Nemotron-3-Nano-30B-A3B-BF16
# Tiempo estimado: 4-6 horas en A100 80GB
# ============================================================

import os, json, torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

# ── Hiperparámetros ──────────────────────────────────────────
MODEL_NAME    = "unsloth/Nemotron-3-Nano-30B-A3B"  # versión optimizada de Unsloth
DATA_PATH     = "/workspace/data/training_data.jsonl"
OUTPUT_DIR    = "/workspace/adapter"
LORA_RANK     = 16
MAX_SEQ_LEN   = 1024
NUM_EPOCHS    = 2
BATCH_SIZE    = 1
GRAD_ACCUM    = 4
LR            = 1e-4

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Cargar datos ─────────────────────────────────────────────
print("Cargando datos de entrenamiento...")
records = []
with open(DATA_PATH) as f:
    for line in f:
        records.append(json.loads(line))
print(f"✅ {len(records)} ejemplos cargados")

# ── Cargar modelo con Unsloth ────────────────────────────────
print("Cargando modelo (puede tardar ~10 min en descargar)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
    dtype=None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=LORA_RANK,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=False,
)
model.print_trainable_parameters()

# ── Preparar dataset ─────────────────────────────────────────
def format_example(example):
    user_content      = example["messages"][0]["content"]
    assistant_content = example["messages"][1]["content"]
    user_msg = user_content + "\nPut your final answer inside \\boxed{}."
    messages = [
        {"role": "user",      "content": user_msg},
        {"role": "assistant", "content": assistant_content},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}

hf_dataset = Dataset.from_list(records)
hf_dataset = hf_dataset.map(format_example, remove_columns=hf_dataset.column_names)
print(f"✅ Dataset preparado: {len(hf_dataset)} ejemplos")

# ── Configuración de entrenamiento ───────────────────────────
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    logging_steps=10,
    bf16=True,
    max_grad_norm=1.0,
    optim="adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    save_strategy="no",
    report_to="none",
    dataset_text_field="text",
    max_length=MAX_SEQ_LEN,
    packing=False,  # Mejora eficiencia
    gradient_checkpointing=False,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

trainer = SFTTrainer(
    model=model,
    train_dataset=hf_dataset,
    processing_class=tokenizer,
    args=training_args,
)

# ── Entrenar ─────────────────────────────────────────────────
print("=== Iniciando entrenamiento ===")
trainer.train()

# ── Guardar adapter ──────────────────────────────────────────
print("Guardando adapter...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n✅ Adapter guardado en {OUTPUT_DIR}:")
for f in os.listdir(OUTPUT_DIR):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    print(f"  {f} ({size/1024:.1f} KB)")
print("=== Entrenamiento completado ===")
