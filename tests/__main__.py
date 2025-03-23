"""
Main test runner module for Fobos SDR wrapper tests.
Run all tests with: python -m tests
"""

import unittest
import argparse
import sys
import logging

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Fobos SDR wrapper tests')
    parser.add_argument('--verbose', '-v', action='store_true', 
                        help='Enable verbose output')
    parser.add_argument('--integration', '-i', action='store_true',
                        help='Run integration tests (requires hardware)')
    parser.add_argument('--no-mock', action='store_true',
                        help='Skip mock tests')
    parser.add_argument('--no-logic', action='store_true',
                        help='Skip wrapper logic tests')
    return parser.parse_args()

def run_tests():
    """Run the test suite."""
    args = parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Load test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test modules based on arguments
    if not args.no_mock:
        from tests import test_mock_fobos
        suite.addTest(loader.loadTestsFromModule(test_mock_fobos))
        
    if not args.no_logic:
        from tests import test_wrapper_logic
        suite.addTest(loader.loadTestsFromModule(test_wrapper_logic))
        
    if args.integration:
        from tests import test_integration
        suite.addTest(loader.loadTestsFromModule(test_integration))
    
    # Run the tests
    verbosity = 2 if args.verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Return appropriate exit code
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    sys.exit(run_tests())