#!/usr/bin/env python3
"""
WandB Troubleshooting Script for InstructPix2Pix

This script helps diagnose common issues with WandB integration.
Run this before training to ensure WandB is properly configured.
"""

import os
import sys
import argparse

def check_wandb_environment():
    """Check environment variables related to WandB"""
    print("\n=== Checking WandB Environment Variables ===")
    
    api_key = os.environ.get('WANDB_API_KEY')
    if api_key:
        masked_key = api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]
        print(f"✅ WANDB_API_KEY is set: {masked_key}")
    else:
        print("❌ WANDB_API_KEY is not set. You should set this in your .env file or environment.")
        print("   You can get your API key from https://wandb.ai/settings")
    
    project = os.environ.get('WANDB_PROJECT')
    if project:
        print(f"✅ WANDB_PROJECT is set: {project}")
    else:
        print("ℹ️ WANDB_PROJECT is not set. Will use default ('instruct-pix2pix').")
    
    entity = os.environ.get('WANDB_ENTITY')
    if entity:
        print(f"✅ WANDB_ENTITY is set: {entity}")
    else:
        print("ℹ️ WANDB_ENTITY is not set. Will use your default wandb user.")
    
    mode = os.environ.get('WANDB_MODE')
    if mode:
        print(f"ℹ️ WANDB_MODE is set to: {mode}")
        if mode == "offline":
            print("   Note: WandB is in offline mode. Runs won't be synced to wandb.ai")
    else:
        print("ℹ️ WANDB_MODE is not set. Will use default (online) mode.")

def check_wandb_import():
    """Check if wandb can be imported"""
    print("\n=== Checking WandB Installation ===")
    
    try:
        import wandb
        print(f"✅ WandB is installed: version {wandb.__version__}")
        
        try:
            # Check if API key works
            api = wandb.Api()
            entity = api.default_entity or "[default]"
            print(f"✅ WandB API connection successful (entity: {entity})")
        except wandb.errors.CommError:
            print("❌ Could not connect to WandB API. Check your internet connection.")
        except wandb.errors.AuthenticationError:
            print("❌ WandB API key is invalid or not properly set.")
        except Exception as e:
            print(f"❌ Error when checking WandB API: {str(e)}")
            
        # Check if already logged in
        try:
            if wandb.api.api_key:
                print("✅ WandB login status: Logged in")
            else:
                print("❌ WandB login status: Not logged in")
        except:
            print("❌ Could not determine WandB login status")
            
    except ImportError:
        print("❌ WandB is not installed. Install with: pip install wandb")
        
    except Exception as e:
        print(f"❌ Error when importing WandB: {str(e)}")

def run_test_init():
    """Try to initialize wandb"""
    print("\n=== Testing WandB Initialization ===")
    
    try:
        import wandb
        
        # Try to initialize wandb
        try:
            run = wandb.init(project="instruct-pix2pix-test", name="test-init")
            print(f"✅ WandB initialized successfully: {run.name}")
            run.finish()
        except wandb.errors.UsageError:
            print("❌ WandB initialization failed. Make sure your API key is correct.")
        except Exception as e:
            print(f"❌ Error initializing WandB: {str(e)}")
            
    except ImportError:
        print("❌ WandB is not installed, skipping initialization test.")

def check_accelerate_config():
    """Check accelerate config for wandb integration"""
    print("\n=== Checking Accelerate Configuration ===")
    
    try:
        from accelerate.utils import ProjectConfiguration
        print("✅ Accelerate is installed")
        
        # Try to find accelerate config file
        import os
        home = os.path.expanduser("~")
        config_path = os.path.join(home, ".cache", "huggingface", "accelerate", "default_config.yaml")
        
        if os.path.exists(config_path):
            print(f"✅ Found accelerate config: {config_path}")
            
            # Read config and check for wandb
            with open(config_path, "r") as f:
                config = f.read()
                if "wandb" in config.lower():
                    print("✅ WandB is referenced in accelerate config")
                else:
                    print("❌ WandB is not referenced in accelerate config")
        else:
            print(f"ℹ️ No accelerate config found at {config_path}")
            
    except ImportError:
        print("❌ Accelerate is not installed, skipping config check.")

def main():
    parser = argparse.ArgumentParser(description="Troubleshoot WandB integration")
    parser.add_argument("--full", action="store_true", help="Run all checks including test initialization")
    args = parser.parse_args()

    print("==================================")
    print("WandB Troubleshooting for InstructPix2Pix")
    print("==================================")
    
    check_wandb_environment()
    check_wandb_import()
    check_accelerate_config()
    
    if args.full:
        run_test_init()
        
    print("\n=== Summary ===")
    print("If any issues were found, here's what to do:")
    print("1. Make sure WANDB_API_KEY is set in your environment or .env file")
    print("2. Check that wandb is installed: pip install wandb")
    print("3. Try logging in manually: wandb login")
    print("4. Check your internet connection")
    print("5. Try running in offline mode if needed: export WANDB_MODE=offline")
    
    print("\nSee more help at: https://docs.wandb.ai/guides/track/advanced/environment-variables")

if __name__ == "__main__":
    main()
