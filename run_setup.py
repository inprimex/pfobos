#!/usr/bin/env python3
"""
Fobos SDR Setup Script
Verifies Python environment and dependencies for the Fobos SDR project.
"""

import sys
import os
import subprocess
import platform
from pathlib import Path
from importlib import util

def check_python_version():
    """Check if Python version meets requirements."""
    required_version = (3, 7)
    current_version = sys.version_info[:2]
    
    if current_version < required_version:
        print(f"Error: Python {required_version[0]}.{required_version[1]} or higher is required")
        print(f"Current version: {current_version[0]}.{current_version[1]}")
        return False
    return True

def check_package_installed(package_name):
    """Check if a Python package is installed."""
    return util.find_spec(package_name) is not None

def parse_requirements(filename):
    """Parse requirements.txt and return list of package names."""
    requirements = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Extract package name from requirement line
                # Handle different formats like:
                # package==1.0.0
                # package>=1.0.0
                # package~=1.0.0
                package = line.split('>=')[0].split('==')[0].split('~=')[0].strip()
                requirements.append(package)
    return requirements

def check_dependencies():
    """Check if required packages are installed."""
    try:
        requirements = parse_requirements('requirements.txt')
        missing = []
        
        for package in requirements:
            if not check_package_installed(package):
                missing.append(package)
        
        if missing:
            print("Missing packages:")
            for package in missing:
                print(f"  - {package}")
            return False
        return True
    except Exception as e:
        print(f"Dependencies check failed: {e}")
        return False

def verify_project_structure():
    """Verify that all required project files and directories exist."""
    required_paths = [
        'requirements.txt',
        'rtanalyzer/__init__.py',
        'rtanalyzer/rtanalyzer.py',
        'shared/__init__.py',
        'shared/fwrapper.py',
        'setup/setup-fobos-sdr.sh'
    ]
    
    missing = []
    for path in required_paths:
        if not Path(path).exists():
            missing.append(path)
    
    if missing:
        print("Missing required files:")
        for path in missing:
            print(f"  - {path}")
        return False
    return True

def check_hardware_setup():
    """Check if Fobos SDR hardware is properly configured."""
    # Check udev rules
    udev_path = Path('/etc/udev/rules.d/99-fobos-sdr.rules')
    if not udev_path.exists():
        print("Warning: Fobos SDR udev rules not found")
        print("Run 'sudo ./setup/setup-fobos-sdr.sh' to configure device permissions")
        return False
    
    # Check if device is connected
    try:
        result = subprocess.run(['lsusb'], capture_output=True, text=True)
        if '16d0:132e' in result.stdout:
            print("Fobos SDR device detected")
            return True
        else:
            print("Warning: Fobos SDR device not detected")
            print("Make sure the device is connected and properly recognized")
            return False
    except Exception as e:
        print(f"Error checking hardware: {e}")
        return False

def main():
    """Main setup verification function."""
    print("Verifying Fobos SDR setup...")
    
    # Track overall status
    status = True
    
    # Check Python version
    print("\nChecking Python version...")
    if not check_python_version():
        status = False
    
    # Verify project structure
    print("\nVerifying project structure...")
    if not verify_project_structure():
        status = False
    
    # Check dependencies
    print("\nChecking Python dependencies...")
    if not check_dependencies():
        print("Run 'pip install -r requirements.txt' to install required packages")
        status = False
    
    # Check hardware setup
    print("\nChecking hardware setup...")
    if not check_hardware_setup():
        status = False
    
    # Print summary
    print("\nSetup verification complete!")
    if status:
        print("All checks passed successfully!")
    else:
        print("Some checks failed. Please address the issues mentioned above.")
        sys.exit(1)

if __name__ == '__main__':
    main()