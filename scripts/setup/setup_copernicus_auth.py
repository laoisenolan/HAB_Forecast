#!/usr/bin/env python3
"""
Copernicus Marine Authentication Setup Script

This script sets up proper authentication for the copernicusmarine library
using multiple methods to ensure compatibility.
"""

import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

def setup_copernicus_auth():
    """Set up Copernicus Marine authentication using multiple methods"""
    
    # Load environment variables
    load_dotenv()
    
    username = os.getenv("COPERNICUS_USERNAME")
    password = os.getenv("COPERNICUS_PASSWORD")
    
    if not username or not password:
        print("❌ Copernicus Marine credentials not found in .env file")
        print("Please set COPERNICUS_USERNAME and COPERNICUS_PASSWORD")
        return False
    
    print(f"🔑 Setting up Copernicus Marine authentication for user: {username}")
    
    # Method 1: Use copernicusmarine CLI login
    try:
        print("📝 Method 1: Using copernicusmarine CLI login...")
        result = subprocess.run([
            sys.executable, "-m", "copernicusmarine", "login",
            "--username", username,
            "--password", password
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ CLI login successful")
        else:
            print(f"⚠️ CLI login failed: {result.stderr}")
    except Exception as e:
        print(f"⚠️ CLI login error: {e}")
    
    # Method 2: Set environment variables
    print("📝 Method 2: Setting environment variables...")
    os.environ["COPERNICUS_USERNAME"] = username
    os.environ["COPERNICUS_PASSWORD"] = password
    print("✅ Environment variables set")
    
    # Method 3: Create config files
    print("📝 Method 3: Creating config files...")
    
    # Home directory config
    home_config_dir = Path.home() / ".copernicusmarine"
    home_config_dir.mkdir(exist_ok=True)
    
    home_config_file = home_config_dir / "credentials"
    with open(home_config_file, 'w') as f:
        f.write(f"username={username}\n")
        f.write(f"password={password}\n")
    print(f"✅ Home config created: {home_config_file}")
    
    # Current directory config
    current_config_dir = Path.cwd() / ".copernicusmarine"
    current_config_dir.mkdir(exist_ok=True)
    
    current_config_file = current_config_dir / "credentials"
    with open(current_config_file, 'w') as f:
        f.write(f"username={username}\n")
        f.write(f"password={password}\n")
    print(f"✅ Current directory config created: {current_config_file}")
    
    # Method 4: Test authentication
    print("📝 Method 4: Testing authentication...")
    try:
        from copernicusmarine import subset
        
        # Try a small test download
        print("🧪 Testing with a small data subset...")
        test_result = subset(
            dataset_id="GLOBAL_ANALYSISFORECAST_PHY_001_024",
            variables=["thetao"],
            minimum_longitude=132.0,
            maximum_longitude=132.1,
            minimum_latitude=34.0,
            maximum_latitude=34.1,
            start_datetime="2024-01-01",
            end_datetime="2024-01-01",
            output_directory="./test_output",
            output_filename="test_auth.nc"
        )
        print("✅ Authentication test successful!")
        
        # Clean up test file
        test_file = Path("./test_output/test_auth.nc")
        if test_file.exists():
            test_file.unlink()
        test_dir = Path("./test_output")
        if test_dir.exists():
            test_dir.rmdir()
            
    except Exception as e:
        print(f"❌ Authentication test failed: {e}")
        return False
    
    print("🎉 Copernicus Marine authentication setup completed successfully!")
    return True

if __name__ == "__main__":
    success = setup_copernicus_auth()
    if success:
        print("\n🚀 You can now run the data collection:")
        print("python src/data_collection/main.py collect-weekly")
    else:
        print("\n❌ Authentication setup failed. Please check your credentials.")
        sys.exit(1) 