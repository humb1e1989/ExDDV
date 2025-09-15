#!/usr/bin/env python3
"""
Script to check LLaVA version and related environment information
"""

import sys
import torch
import importlib
import subprocess
import json
from pathlib import Path

def check_package_version(package_name):
    """Check version of a Python package"""
    try:
        module = importlib.import_module(package_name)
        if hasattr(module, '__version__'):
            return module.__version__
        elif hasattr(module, 'VERSION'):
            return module.VERSION
        else:
            # Try using pip show as fallback
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'show', package_name],
                capture_output=True,
                text=True
            )
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    return line.split(':')[1].strip()
            return "Version not found"
    except ImportError:
        return "Not installed"
    except Exception as e:
        return f"Error: {str(e)}"

def check_llava_info():
    """Check LLaVA specific information"""
    print("=" * 60)
    print("LLaVA Environment Information")
    print("=" * 60)
    
    # 1. Check if LLaVA is installed
    try:
        import llava
        print("✓ LLaVA is installed")
        
        # Check LLaVA version
        if hasattr(llava, '__version__'):
            print(f"  LLaVA version: {llava.__version__}")
        else:
            print("  LLaVA version: Not specified in module")
            
        # Check installation path
        llava_path = Path(llava.__file__).parent
        print(f"  Installation path: {llava_path}")
        
        # Check if it's a git repository (development install)
        git_dir = llava_path.parent / '.git'
        if git_dir.exists():
            try:
                # Get git commit hash
                result = subprocess.run(
                    ['git', 'rev-parse', 'HEAD'],
                    cwd=llava_path.parent,
                    capture_output=True,
                    text=True
                )
                commit_hash = result.stdout.strip()[:8]
                print(f"  Git commit: {commit_hash}")
                
                # Get git branch
                result = subprocess.run(
                    ['git', 'branch', '--show-current'],
                    cwd=llava_path.parent,
                    capture_output=True,
                    text=True
                )
                branch = result.stdout.strip()
                print(f"  Git branch: {branch}")
            except:
                print("  Git info: Unable to retrieve")
        
        # Check available components
        components = []
        try:
            import llava.model
            components.append("model")
        except: pass
        
        try:
            import llava.mm_utils
            components.append("mm_utils")
        except: pass
        
        try:
            import llava.conversation
            components.append("conversation")
        except: pass
        
        try:
            import llava.constants
            components.append("constants")
        except: pass
        
        print(f"  Available components: {', '.join(components)}")
        
    except ImportError as e:
        print(f"✗ LLaVA is not installed or not found in Python path")
        print(f"  Error: {e}")
        print("\n  To install LLaVA:")
        print("  git clone https://github.com/haotian-liu/LLaVA.git")
        print("  cd LLaVA")
        print("  pip install -e .")
    
    print()

def check_model_info(model_path="liuhaotian/llava-v1.5-7b"):
    """Check model specific information"""
    print("=" * 60)
    print("Model Information")
    print("=" * 60)
    
    try:
        from transformers import AutoConfig
        
        # Try to load model config from HuggingFace
        print(f"Checking model: {model_path}")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        print(f"  Model type: {config.model_type if hasattr(config, 'model_type') else 'Unknown'}")
        print(f"  Hidden size: {config.hidden_size if hasattr(config, 'hidden_size') else 'Unknown'}")
        print(f"  Num layers: {config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else 'Unknown'}")
        
        # Check for LLaVA specific configs
        if hasattr(config, 'mm_vision_tower'):
            print(f"  Vision tower: {config.mm_vision_tower}")
        if hasattr(config, 'mm_hidden_size'):
            print(f"  MM hidden size: {config.mm_hidden_size}")
            
    except Exception as e:
        print(f"  Unable to load model config: {e}")
    
    print()

def check_dependencies():
    """Check all related dependencies"""
    print("=" * 60)
    print("Dependencies Version Information")
    print("=" * 60)
    
    packages = [
        'torch',
        'torchvision',
        'transformers',
        'tokenizers',
        'accelerate',
        'pillow',
        'opencv-python',
        'numpy',
        'pandas',
        'clip',
        'sentencepiece',
        'protobuf',
        'peft',
        'bitsandbytes',
    ]
    
    for package in packages:
        # Convert package name for import (e.g., opencv-python -> cv2)
        import_name = package
        if package == 'opencv-python':
            import_name = 'cv2'
        elif package == 'pillow':
            import_name = 'PIL'
        elif package == 'clip':
            import_name = 'clip'
            
        version = check_package_version(import_name)
        status = "✓" if version != "Not installed" else "✗"
        print(f"  {status} {package:20s}: {version}")
    
    print()

def check_cuda_info():
    """Check CUDA and GPU information"""
    print("=" * 60)
    print("CUDA/GPU Information")
    print("=" * 60)
    
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  cuDNN version: {torch.backends.cudnn.version()}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\n  GPU {i}: {props.name}")
            print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"    Compute Capability: {props.major}.{props.minor}")
    else:
        print("  Running on CPU")
    
    print()

def check_llava_model_files(model_path="liuhaotian/llava-v1.5-7b"):
    """Check if model files are cached locally"""
    print("=" * 60)
    print("Model Cache Information")
    print("=" * 60)
    
    from transformers import AutoModel
    from pathlib import Path
    import os
    
    # Check HuggingFace cache
    cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
    print(f"  HuggingFace cache dir: {cache_dir}")
    
    # Look for model files
    model_id = model_path.replace('/', '--')
    model_dirs = list(cache_dir.glob(f"models--{model_id}*"))
    
    if model_dirs:
        print(f"  Found cached model directories:")
        for model_dir in model_dirs:
            print(f"    - {model_dir.name}")
            
            # Check snapshots
            snapshots_dir = model_dir / 'snapshots'
            if snapshots_dir.exists():
                snapshots = list(snapshots_dir.iterdir())
                if snapshots:
                    latest_snapshot = snapshots[-1]
                    print(f"      Latest snapshot: {latest_snapshot.name}")
                    
                    # List model files
                    model_files = list(latest_snapshot.glob("*.bin")) + \
                                 list(latest_snapshot.glob("*.safetensors"))
                    if model_files:
                        print(f"      Model files:")
                        for f in model_files[:3]:  # Show first 3 files
                            size_gb = f.stat().st_size / (1024**3)
                            print(f"        - {f.name} ({size_gb:.2f} GB)")
                        if len(model_files) > 3:
                            print(f"        ... and {len(model_files)-3} more files")
    else:
        print(f"  No cached model found for {model_path}")
        print(f"  Model will be downloaded on first use")
    
    print()

def main():
    """Run all checks"""
    print("\n" + "=" * 60)
    print(" LLaVA ENVIRONMENT CHECK ".center(60))
    print("=" * 60 + "\n")
    
    # Run all checks
    check_llava_info()
    check_dependencies()
    check_cuda_info()
    check_model_info()
    check_llava_model_files()
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    
    # Quick compatibility check
    issues = []
    
    try:
        import llava
    except:
        issues.append("LLaVA not installed")
    
    try:
        import transformers
        from packaging import version
        if version.parse(transformers.__version__) < version.parse("4.31.0"):
            issues.append(f"Transformers version {transformers.__version__} may be too old")
    except:
        issues.append("Transformers not installed")
    
    if not torch.cuda.is_available():
        issues.append("CUDA not available - will run on CPU (slower)")
    
    if issues:
        print("  ⚠ Potential issues found:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  ✓ Environment looks good!")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()