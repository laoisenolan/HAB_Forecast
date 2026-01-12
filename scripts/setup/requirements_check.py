#!/usr/bin/env python3
"""
Simple Requirements Check for HAB Forecasting System

This script checks what's needed to run the data collection system.
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version"""
    print(f"Python version: {sys.version}")
    if sys.version_info >= (3, 8):
        print("✅ Python version is compatible (3.8+)")
        return True
    else:
        print("❌ Python version too old, need 3.8+")
        return False

def check_env_file():
    """Check if .env file exists and has credentials"""
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file found")
        
        # Read and check credentials
        with open(env_file, 'r') as f:
            content = f.read()
            
        if "COPERNICUS_USERNAME" in content and "your_username_here" not in content:
            print("✅ Copernicus Marine credentials found")
        else:
            print("❌ Copernicus Marine credentials missing")
            
        if "COPERNICUS_PASSWORD" in content and "your_password_here" not in content:
            print("✅ Copernicus Marine password found")
        else:
            print("❌ Copernicus Marine password missing")
            
        if "GOOGLE_APPLICATION_CREDENTIALS" in content:
            print("✅ Google Earth Engine credentials path found")
        else:
            print("❌ Google Earth Engine credentials missing")
            
        return True
    else:
        print("❌ .env file not found")
        print("Create .env file with your credentials")
        return False

def check_config_files():
    """Check if configuration files exist"""
    config_dir = Path("config")
    required_files = ["data_sources.yaml", "model_params.yaml", "processing.yaml"]
    
    all_exist = True
    for file in required_files:
        file_path = config_dir / file
        if file_path.exists():
            print(f"✅ {file} found")
        else:
            print(f"❌ {file} missing")
            all_exist = False
            
    return all_exist

def check_directories():
    """Check if required directories exist"""
    required_dirs = [
        "data/raw", "data/processed", "data/climatology", "data/labels",
        "src/data_collection", "src/data_processing", "src/feature_engineering",
        "src/model", "src/evaluation", "src/visualization", "src/utils"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path} exists")
        else:
            print(f"❌ {dir_path} missing")
            all_exist = False
            
    return all_exist

def main():
    """Run all checks"""
    print("🔍 HAB Forecasting System - Requirements Check")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Environment File", check_env_file),
        ("Configuration Files", check_config_files),
        ("Directory Structure", check_directories),
    ]
    
    results = {}
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        results[check_name] = check_func()
    
    print("\n" + "=" * 50)
    print("📋 SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! System is ready for testing.")
        print("\nNext steps:")
        print("1. Install required packages: pip install -r requirements.txt")
        print("2. Test data collection: python src/data_collection/main.py info")
    else:
        print("\n⚠️ Some checks failed. Please address the issues above.")
        print("\nRequired setup:")
        print("1. Install Python 3.8+ if not already installed")
        print("2. Create .env file with your credentials")
        print("3. Run setup script: python setup_environment.py")
        print("4. Install packages: pip install -r requirements.txt")

if __name__ == "__main__":
    main() 