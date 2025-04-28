"""
Main test runner module for Fobos SDR wrapper tests.
Run all tests with: python -m tests
"""

import unittest
import argparse
import sys
import logging
import os
import time
import importlib.util

class StatusTestResult(unittest.TextTestResult):
    """Custom test result class that provides detailed status reporting."""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.stream = stream
        self.passed_tests = []
        self.start_times = {}
        self.verbosity = verbosity  # Store verbosity for later use
        
    def startTest(self, test):
        self.start_times[test] = time.time()
        super().startTest(test)
        if self.verbosity > 1:
            self.stream.write(f"Running: {test.id()}")
            self.stream.flush()
        
    def addSuccess(self, test):
        super().addSuccess(test)
        duration = time.time() - self.start_times[test]
        self.passed_tests.append((test, duration))
        if self.verbosity > 1:
            self.stream.write(" ... PASS\n")
        else:
            self.stream.write(".")
            
    def addError(self, test, err):
        super().addError(test, err)
        if self.verbosity > 1:
            self.stream.write(" ... ERROR\n")
        else:
            self.stream.write("E")
            
    def addFailure(self, test, err):
        super().addFailure(test, err)
        if self.verbosity > 1:
            self.stream.write(" ... FAIL\n")
        else:
            self.stream.write("F")
            
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        if self.verbosity > 1:
            self.stream.write(f" ... SKIP: {reason}\n")
        else:
            self.stream.write("s")
            
    def printSummary(self):
        """Print a summary of test results."""
        self.stream.writeln("\nTest Results Summary:")
        self.stream.writeln("-" * 70)
        
        # Print passed tests
        if self.passed_tests:
            self.stream.writeln("\nPassed Tests:")
            for test, duration in self.passed_tests:
                self.stream.writeln(f"✓ {test.id()} ({duration:.3f}s)")
                
        # Print errors
        if self.errors:
            self.stream.writeln("\nErrors:")
            for test, _ in self.errors:
                self.stream.writeln(f"✗ {test.id()} - ERROR")
                
        # Print failures
        if self.failures:
            self.stream.writeln("\nFailures:")
            for test, _ in self.failures:
                self.stream.writeln(f"✗ {test.id()} - FAIL")
                
        # Print skipped
        if self.skipped:
            self.stream.writeln("\nSkipped:")
            for test, reason in self.skipped:
                self.stream.writeln(f"⚠ {test.id()} - {reason}")
                
        # Print summary counts
        self.stream.writeln("\nSummary:")
        self.stream.writeln(f"  Total: {self.testsRun}")
        self.stream.writeln(f"  Passed: {len(self.passed_tests)}")
        self.stream.writeln(f"  Failed: {len(self.failures)}")
        self.stream.writeln(f"  Errors: {len(self.errors)}")
        self.stream.writeln(f"  Skipped: {len(self.skipped)}")
        

class StatusTestRunner(unittest.TextTestRunner):
    """Custom test runner that uses StatusTestResult."""
    
    resultclass = StatusTestResult
    
    def run(self, test):
        result = super().run(test)
        result.printSummary()
        return result


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
    parser.add_argument('--performance-only', '-p', action='store_true',
                        help='Run only performance tests (requires hardware)')
    parser.add_argument('--benchmark', '-b', action='store_true',
                        help='Run the benchmark tool instead of unit tests')
    parser.add_argument('--device', type=int, default=0,
                        help='Device index to use for benchmark (default: 0)')
    parser.add_argument('--iterations', type=int, default=3,
                        help='Number of iterations for benchmark (default: 3)')
    parser.add_argument('--output-dir', type=str, default="benchmark_results",
                        help='Directory to save benchmark results (default: benchmark_results)')
    parser.add_argument('--plot-only', action='store_true',
                        help='Only generate plots from existing benchmark results')
    return parser.parse_args()

def run_benchmark(args):
    """Run the benchmark tool with the provided arguments."""
    # Find benchmark.py in the tests directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    benchmark_path = os.path.join(project_root, 'tests', 'benchmark.py')
    
    if not os.path.exists(benchmark_path):
        print(f"Error: benchmark.py not found at {benchmark_path}")
        return 1
        
    print(f"Running benchmark from {benchmark_path}")
    
    # Load the benchmark module
    spec = importlib.util.spec_from_file_location("benchmark", benchmark_path)
    benchmark_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark_module)
    
    # Create a FobosSDRBenchmark instance with the provided arguments
    benchmark = benchmark_module.FobosSDRBenchmark(args.device, args.output_dir)
    
    if args.plot_only:
        # Just generate comparison plots from existing results
        benchmark.generate_comparison_plots()
        print("Benchmark plots generated successfully")
        return 0
    else:
        # Run all benchmarks
        success = benchmark.run_all_benchmarks(args.iterations)
        
        if success:
            # Also generate comparison plots
            benchmark.generate_comparison_plots()
            print("Benchmarking completed successfully")
            return 0
        else:
            print("Benchmarking failed")
            return 1

def run_tests():
    """Run the test suite."""
    args = parse_args()
    
    # Check if we should run the benchmark tool instead of unit tests
    if args.benchmark:
        return run_benchmark(args)
    
    # Configure logging level
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Add project root to sys.path to allow imports to work correctly
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    # Load test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Handle performance-only case
    if args.performance_only:
        from tests import test_performance
        suite.addTest(loader.loadTestsFromModule(test_performance))
        print("Running performance tests only (requires hardware)")
    else:
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
            
            # Also include performance tests when integration is enabled
            from tests import test_performance
            suite.addTest(loader.loadTestsFromModule(test_performance))
    
    # Run the tests with our custom runner
    verbosity = 2 if args.verbose else 1
    runner = StatusTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Return appropriate exit code
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    sys.exit(run_tests())