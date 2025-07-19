#!/usr/bin/env python3
"""
Display startup configuration information
"""

import os
from pathlib import Path
from config import get_settings, CONFIG_DIR, DEVICE_CONFIG_FILE

def print_startup_info():
    """Print configuration information at startup"""
    print("=" * 60)
    print("NoEmbed Service - Startup Configuration")
    print("=" * 60)
    
    # Check config directory
    print(f"\nConfiguration Directory: {CONFIG_DIR}")
    print(f"  - Exists: {CONFIG_DIR.exists()}")
    print(f"  - Writable: {os.access(CONFIG_DIR, os.W_OK) if CONFIG_DIR.exists() else 'N/A'}")
    
    # Check device config file
    print(f"\nDevice Config File: {DEVICE_CONFIG_FILE}")
    print(f"  - Exists: {DEVICE_CONFIG_FILE.exists()}")
    
    if DEVICE_CONFIG_FILE.exists():
        try:
            with open(DEVICE_CONFIG_FILE, 'r') as f:
                content = f.read()
                print(f"  - Content: {content.strip()}")
        except Exception as e:
            print(f"  - Error reading: {e}")
    
    # Check .env file
    env_file = Path(".env")
    print(f"\n.env File: {env_file}")
    print(f"  - Exists: {env_file.exists()}")
    
    if env_file.exists():
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    if 'DEVICE' in line:
                        print(f"  - Device setting: {line.strip()}")
                        break
        except Exception as e:
            print(f"  - Error reading: {e}")
    
    # Get final settings
    settings = get_settings()
    print(f"\nFinal Configuration:")
    print(f"  - Device: {settings.device}")
    print(f"  - Model: {settings.model_name}")
    print(f"  - Port: {settings.port}")
    print(f"  - Max Batch Size: {settings.max_batch_size}")
    
    print("\nConfiguration Priority:")
    print("  1. config/device.json (if exists)")
    print("  2. .env file")
    print("  3. Default values")
    
    print("=" * 60)

if __name__ == "__main__":
    print_startup_info()