"""
Run All MLflow Experiments

Orchestrates running all experiment types in test mode.
Provides a comprehensive validation of the entire framework.

Usage:
    python run_all_experiments.py              # Run all in test mode
    python run_all_experiments.py --full       # Run all experiments fully
    python run_all_experiments.py --ui         # Start MLflow UI after completion
"""

import subprocess
import sys
import os
from pathlib import Path
import time
import argparse

# ANSI color codes for better output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print a formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(70)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}={'='*70}{Colors.ENDC}\n")


def print_success(text):
    """Print success message."""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_error(text):
    """Print error message."""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_info(text):
    """Print info message."""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


def check_dependencies():
    """Check if required packages are installed."""
    print_info("Checking dependencies...")
    
    required_packages = [
        'mlflow',
        'sentence_transformers',
        'rank_bm25',
        'numpy',
        'matplotlib',
        'sklearn'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print_error(f"Missing packages: {', '.join(missing)}")
        print_info("Installing missing packages...")
        subprocess.run([
            sys.executable, '-m', 'pip', 'install',
            'mlflow', 'sentence-transformers', 'rank-bm25', 
            'pynvml', 'matplotlib', 'scikit-learn'
        ])
    else:
        print_success("All dependencies installed")


def check_ollama_for_llm():
    """Check if Ollama is available for LLM experiments."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print_success("Ollama server is running")
            return True
        else:
            print_error("Ollama server not responding correctly")
            return False
    except:
        print_error("Ollama server not available")
        print_info("LLM experiments will be skipped")
        print_info("To enable: Run 'ollama serve' in another terminal")
        return False


def run_experiment(name, directory, script, args=None, skip=False):
    """
    Run a single experiment.
    
    Args:
        name: Experiment name for display
        directory: Directory containing the experiment
        script: Script name to run
        args: Additional arguments for the script
        skip: If True, skip this experiment
        
    Returns:
        True if successful, False otherwise
    """
    if skip:
        print_info(f"Skipping {name}")
        return True
    
    print(f"\n{Colors.BOLD}Running: {name}{Colors.ENDC}")
    print("-" * 70)
    
    script_path = Path(__file__).parent / directory / script
    
    if not script_path.exists():
        print_error(f"Script not found: {script_path}")
        return False
    
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent / directory,
            capture_output=False,
            text=True
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print_success(f"{name} completed in {elapsed:.1f}s")
            return True
        else:
            print_error(f"{name} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print_error(f"{name} failed with error: {e}")
        return False


def start_mlflow_ui():
    """Start MLflow UI."""
    print_header("Starting MLflow UI")
    print_info("MLflow UI will be available at: http://localhost:5000")
    print_info("Press Ctrl+C to stop the server")
    print()
    
    try:
        # Construct URI
        db_path = Path(__file__).parent / "mlflow.db"
        db_uri = f"sqlite:///{db_path.as_posix()}"
        
        cmd = [sys.executable, '-m', 'mlflow', 'ui', '--backend-store-uri', db_uri]
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n")
        print_info("MLflow UI stopped")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run all MLflow experiments")
    parser.add_argument('--full', action='store_true', 
                       help='Run full experiments (not test mode)')
    parser.add_argument('--ui', action='store_true',
                       help='Start MLflow UI after experiments')
    parser.add_argument('--skip-llm', action='store_true',
                       help='Skip LLM experiments (useful if Ollama not available)')
    args = parser.parse_args()
    
    print_header("MLflow Experiments - Full Suite")
    
    # Check dependencies
    check_dependencies()
    
    # Check Ollama availability
    ollama_available = check_ollama_for_llm()
    skip_llm = args.skip_llm or not ollama_available
    
    # Prepare experiment configurations
    test_args = [] if args.full else ['--test-mode']
    
    experiments = [
        {
            'name': '1. Embedding Model Experiments',
            'directory': 'embedding_experiments',
            'script': 'run_experiments.py',
            'args': test_args,
            'skip': False
        },
        {
            'name': '2. Retrieval Strategy Experiments',
            'directory': 'retrieval_experiments',
            'script': 'run_experiments.py',
            'args': test_args,
            'skip': False
        },
        {
            'name': '3. Chunking Strategy Experiments',
            'directory': 'chunking_experiments',
            'script': 'run_experiments.py',
            'args': test_args,
            'skip': False
        },
        {
            'name': '4. LLM Model Experiments',
            'directory': 'llm_experiments',
            'script': 'run_experiments.py',
            'args': test_args + ['--num-queries', '3'],
            'skip': skip_llm
        }
    ]
    
    # Track results
    results = {}
    start_time = time.time()
    
    # Run experiments
    print_header("Running Experiments")
    
    for exp_config in experiments:
        success = run_experiment(
            exp_config['name'],
            exp_config['directory'],
            exp_config['script'],
            exp_config['args'],
            exp_config['skip']
        )
        results[exp_config['name']] = success
    
    total_time = time.time() - start_time
    
    # Print summary
    print_header("Experiment Summary")
    
    passed = sum(1 for success in results.values() if success)
    total = len([r for r in results.values() if r is not None])
    
    for name, success in results.items():
        if success is None:
            continue
        if success:
            print_success(f"{name}")
        else:
            print_error(f"{name}")
    
    print()
    print(f"{Colors.BOLD}Total: {passed}/{total} experiments passed{Colors.ENDC}")
    print(f"{Colors.BOLD}Total time: {total_time:.1f}s ({total_time/60:.1f} minutes){Colors.ENDC}")
    
    if passed == total:
        print()
        print_success("All experiments completed successfully! 🎉")
        print()
        print_info("Next steps:")
        print_info("  1. View results: mlflow ui")
        print_info("  2. Navigate to: http://localhost:5000")
        print_info("  3. Compare runs and generate charts for your thesis")
        
        if args.ui:
            print()
            input("Press Enter to start MLflow UI...")
            start_mlflow_ui()
    else:
        print()
        print_error("Some experiments failed. Check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n")
        print_info("Experiments interrupted by user")
        sys.exit(0)
    except Exception as e:
        print()
        print_error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
