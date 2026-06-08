#!/bin/bash
# ============================================================
# Script 1: Instalación de dependencias para RunPod A100
# Tiempo estimado: ~5 minutos
# ============================================================

echo "=== Instalando dependencias ==="

# Unsloth con soporte Nemotron
pip install unsloth unsloth_zoo -q

# Mamba-ssm y causal-conv1d (versiones específicas que funcionan)
pip install --no-build-isolation mamba_ssm==2.2.5
pip install --no-build-isolation causal_conv1d==1.5.2

# Librerías de entrenamiento
pip install trl peft bitsandbytes datasets -q

# Verificar instalación
python3 -c "
import torch
import unsloth
import mamba_ssm
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA disponible: {torch.cuda.is_available()}')
print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
print(f'✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print(f'✅ Unsloth instalado')
print(f'✅ mamba_ssm instalado')
print('=== Todo listo para entrenar ===')
"
