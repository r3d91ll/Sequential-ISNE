#!/usr/bin/env python3
"""
Sequential-ISNE Environment Setup Helper

This script helps users set up their environment for Sequential-ISNE training,
including wandb configuration and dependency verification.
"""

import os
import sys
from pathlib import Path
import subprocess

def check_poetry():
    """Check if Poetry is available."""
    try:
        result = subprocess.run(['poetry', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Poetry found: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ Poetry not found. Please install Poetry first:")
    print("   curl -sSL https://install.python-poetry.org | python3 -")
    return False

def setup_env_file():
    """Set up .env file from template."""
    env_file = Path('.env')
    template_file = Path('.env.template')
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if not template_file.exists():
        print("❌ .env.template not found")
        return False
    
    # Copy template to .env
    template_content = template_file.read_text()
    env_file.write_text(template_content)
    print("✅ Created .env file from template")
    print("📝 Please edit .env file and add your WANDB_API_KEY")
    print("   Get your API key from: https://wandb.ai/authorize")
    return True

def install_dependencies():
    """Install Poetry dependencies."""
    if not check_poetry():
        return False
    
    print("📦 Installing dependencies...")
    
    # Basic installation
    result = subprocess.run(['poetry', 'install'], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Failed to install basic dependencies: {result.stderr}")
        return False
    
    print("✅ Basic dependencies installed")
    
    # Ask user about extras
    extras = []
    
    gpu_support = input("Install GPU support (PyTorch, sentence-transformers)? [y/N]: ").lower().startswith('y')
    if gpu_support:
        extras.append('gpu')
    
    doc_support = input("Install document processing (Docling for PDFs)? [y/N]: ").lower().startswith('y')
    if doc_support:
        extras.append('docs')
    
    if extras:
        extras_str = ','.join(extras)
        print(f"📦 Installing extras: {extras_str}")
        result = subprocess.run(['poetry', 'install', '--extras', extras_str], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"⚠️ Some extras failed to install: {result.stderr}")
            print("You can continue with basic functionality")
        else:
            print(f"✅ Installed extras: {extras_str}")
    
    return True

def verify_wandb():
    """Verify wandb configuration."""
    env_file = Path('.env')
    if not env_file.exists():
        print("⚠️ No .env file found. wandb logging will be disabled.")
        return False
    
    # Check if API key is set
    env_content = env_file.read_text()
    if 'WANDB_API_KEY=your_wandb_api_key_here' in env_content:
        print("⚠️ Please update WANDB_API_KEY in .env file")
        print("   Get your API key from: https://wandb.ai/authorize")
        return False
    
    if 'WANDB_API_KEY=' in env_content:
        print("✅ WANDB_API_KEY found in .env file")
        return True
    
    print("⚠️ WANDB_API_KEY not found in .env file")
    return False

def test_import():
    """Test that key modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        # Test basic Python imports
        import numpy
        import networkx
        print("✅ Core dependencies (numpy, networkx) working")
    except ImportError as e:
        print(f"❌ Core dependency missing: {e}")
        return False
    
    try:
        import wandb
        print("✅ wandb available")
    except ImportError:
        print("⚠️ wandb not available - install with: poetry add wandb")
    
    try:
        from dotenv import load_dotenv
        print("✅ python-dotenv available")
    except ImportError:
        print("⚠️ python-dotenv not available - install with: poetry add python-dotenv")
    
    # Test optional dependencies
    try:
        import torch
        print("✅ PyTorch available")
    except ImportError:
        print("ℹ️ PyTorch not available (optional)")
    
    try:
        import sentence_transformers
        print("✅ sentence-transformers available")
    except ImportError:
        print("ℹ️ sentence-transformers not available (optional)")
    
    return True

def main():
    """Run the complete setup process."""
    print("🔬 Sequential-ISNE Environment Setup")
    print("=" * 50)
    
    # Check working directory
    if not Path('pyproject.toml').exists():
        print("❌ Please run this script from the Sequential-ISNE directory")
        sys.exit(1)
    
    success = True
    
    # 1. Check Poetry
    if not check_poetry():
        success = False
    
    # 2. Set up .env file
    if not setup_env_file():
        success = False
    
    # 3. Install dependencies
    if success and not install_dependencies():
        success = False
    
    # 4. Verify wandb setup
    wandb_ok = verify_wandb()
    
    # 5. Test imports
    if success and not test_import():
        success = False
    
    print("\n" + "=" * 50)
    
    if success:
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Edit .env file with your WANDB_API_KEY (if using wandb)")
        print("2. Run training: python train_and_log_model.py")
        
        if not wandb_ok:
            print("\nNote: wandb logging will be disabled without API key")
    else:
        print("❌ Setup encountered some issues")
        print("Please check the errors above and try again")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())