import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
except: print("✗ PyTorch")

try:
    import transformers
    print(f"✓ Transformers: {transformers.__version__}")
except: print("✗ Transformers")

try:
    import numpy as np
    print(f"✓ NumPy: {np.__version__}")
except: print("✗ NumPy")

try:
    import cv2
    print(f"✓ OpenCV: {cv2.__version__}")
except: print("✗ OpenCV")

try:
    import clip
    print(f"✓ CLIP: Installed")
except: print("✗ CLIP")

try:
    import llava
    print(f"✓ LLaVA: Installed")
except: print("✗ LLaVA")