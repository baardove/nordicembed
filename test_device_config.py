#!/usr/bin/env python3
"""
Test device configuration system
"""

import os
import json
import shutil
from pathlib import Path
from config import get_settings, save_device_config, CONFIG_DIR, DEVICE_CONFIG_FILE

def test_device_config():
    print("Testing Device Configuration System")
    print("=" * 50)
    
    # 1. Test with no config file
    print("\n1. Testing with no config file...")
    if DEVICE_CONFIG_FILE.exists():
        os.remove(DEVICE_CONFIG_FILE)
    
    settings = get_settings()
    print(f"   Device from .env: {settings.device}")
    
    # 2. Test saving device config
    print("\n2. Testing save_device_config...")
    success = save_device_config("cuda")
    print(f"   Save result: {'Success' if success else 'Failed'}")
    
    if DEVICE_CONFIG_FILE.exists():
        with open(DEVICE_CONFIG_FILE, 'r') as f:
            print(f"   File content: {f.read().strip()}")
    
    # 3. Test loading with config file
    print("\n3. Testing with config file...")
    # Need to clear cache and reimport
    get_settings.cache_clear()
    from config import get_settings as get_settings_new
    settings = get_settings_new()
    print(f"   Device from config: {settings.device}")
    
    # 4. Test invalid device
    print("\n4. Testing invalid device...")
    success = save_device_config("gpu")  # Invalid
    print(f"   Save invalid device: {'Success' if success else 'Failed (expected)'}")
    
    # 5. Test config override
    print("\n5. Testing configuration priority...")
    print("   Creating config with 'cpu'...")
    save_device_config("cpu")
    
    # Clear cache again
    get_settings.cache_clear()
    settings = get_settings_new()
    print(f"   Final device setting: {settings.device}")
    print(f"   (Should be 'cpu' from config, not '{os.getenv('DEVICE', 'default')}' from .env)")
    
    print("\n" + "=" * 50)
    print("Configuration Priority:")
    print("1. config/device.json (highest priority)")
    print("2. .env file") 
    print("3. Default values (lowest priority)")

if __name__ == "__main__":
    test_device_config()