#!/usr/bin/env python3
"""
Entry point script for the Enhanced Real-Time Spectrum Analyzer
"""

import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the analyzer from the rtanalyzer package
from rtanalyzer import EnhancedRealTimeAnalyzer

if __name__ == "__main__":
    print("Starting Enhanced Real-Time Spectrum Analyzer...")
    
    try:
        analyzer = EnhancedRealTimeAnalyzer()
        analyzer.run()
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()