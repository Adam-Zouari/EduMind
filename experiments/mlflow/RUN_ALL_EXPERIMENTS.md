# Run All Experiments

Master script to run all MLflow experiments sequentially.

## Usage

### Quick Test (Recommended First Run)
```bash
cd mlflow
python run_all_experiments.py
```

This runs all experiments in **test mode** (fewer models/strategies for faster completion).

### Full Experiments
```bash
python run_all_experiments.py --full
```

Runs all experiments with complete model/strategy sets.

### With MLflow UI
```bash
python run_all_experiments.py --ui
```

Automatically starts MLflow UI after experiments complete.

### Skip LLM Experiments
```bash
python run_all_experiments.py --skip-llm
```

Skip LLM experiments if Ollama is not available.

## What It Does

1. **Checks dependencies** - Installs missing packages automatically
2. **Checks Ollama** - Verifies if LLM experiments can run
3. **Runs experiments** in sequence:
   - Embedding experiments (2 models in test mode, 5 in full)
   - Retrieval experiments (2 strategies in test mode, 4 in full)
   - Chunking experiments (2 strategies in test mode, 5 in full)
   - LLM experiments (1 model with 3 queries in test mode, skip if Ollama unavailable)
4. **Shows summary** - Reports which experiments passed/failed
5. **Total time** - Shows how long everything took

## Expected Runtime

- **Test Mode**: ~5-10 minutes (depending on CPU/GPU)
- **Full Mode**: ~15-30 minutes

## Example Output

```
======================================================================
                    MLflow Experiments - Full Suite                    
======================================================================

ℹ Checking dependencies...
✓ All dependencies installed
✓ Ollama server is running

======================================================================
                         Running Experiments                          
======================================================================

Running: 1. Embedding Model Experiments
----------------------------------------------------------------------
...
✓ 1. Embedding Model Experiments completed in 145.2s

Running: 2. Retrieval Strategy Experiments
----------------------------------------------------------------------
...
✓ 2. Retrieval Strategy Experiments completed in 67.8s

======================================================================
                         Experiment Summary                           
======================================================================

✓ 1. Embedding Model Experiments
✓ 2. Retrieval Strategy Experiments
✓ 3. Chunking Strategy Experiments
✓ 4. LLM Model Experiments

Total: 4/4 experiments passed
Total time: 342.5s (5.7 minutes)

✓ All experiments completed successfully! 🎉
```

## Troubleshooting

**Missing dependencies**: Script auto-installs them

**Ollama not available**: Use `--skip-llm` flag

**Out of memory**: Use test mode instead of full mode

**Specific experiment fails**: Run individual experiments to debug:
```bash
cd embedding_experiments
python run_experiments.py --test-mode
```
