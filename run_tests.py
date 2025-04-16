#!/usr/bin/env python3
"""
Test runner script for Fobos SDR Python wrapper.
"""

import sys
import os
import time
import unittest

# Ensure proper path resolution
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Import custom test runner
from tests.__main__ import run_tests

if __name__ == "__main__":
    print(f"Starting Fobos SDR test suite at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    # Run tests and get result code
    exit_code = run_tests()
    
    # Final message
    if exit_code == 0:
        print("\n✅ All tests passed successfully!")
    else:
        print("\n❌ Some tests failed. Check the results above for details.")
        
    sys.exit(exit_code)