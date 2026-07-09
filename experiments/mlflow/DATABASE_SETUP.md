# MLflow Database Backend - Setup Guide

## Overview

MLflow is now configured to use a **SQLite database** instead of file-based storage. This provides better data management, easier querying, and improved reliability.

## Database Location

- **Database file**: `mlflow/mlflow.db`
- **Artifacts**: `mlflow/mlartifacts/`

## Setup (One-Time)

Run the setup script to initialize the database:

```bash
cd mlflow
python setup_mlflow_db.py
```

This creates:
- `mlflow.db` - SQLite database for all experiment metadata
- `mlartifacts/` - Directory for experiment artifacts
- `mlflow_config.py` - Configuration module for automatic setup
- `.mlflow_config` - Environment configuration file

## Using the Database

### Method 1: Import Configuration (RECOMMENDED)

Add this to the **top** of your experiment scripts:

```python
# Add at the very beginning
from mlflow_config import configure_mlflow
configure_mlflow()

# Then use MLflow normally
import mlflow
from utils import MLflowExperiment

with MLflowExperiment("my_experiment", "my_run") as exp:
    exp.log_params({"param": "value"})
    exp.log_metrics({"metric": 0.95})
```

**✅ This is already done for you** in all experiment scripts!

### Method 2: Environment Variable

Set before running experiments:

```bash
# Windows
set MLFLOW_TRACKING_URI=sqlite:///c:/Users/ademz/Desktop/9raya/MLOps/Project/mlflow/mlflow.db

# Then run experiments
python run_experiments.py
```

### Method 3: Direct in Code

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
```

## Starting MLflow UI

With database backend:

```bash
cd mlflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Or simply:

```bash
cd mlflow
mlflow ui
```

Then open: **http://localhost:5000**

## Benefits of Database Backend

✅ **Better data integrity** - ACID compliance  
✅ **Easier querying** - SQL queries for analysis  
✅ **Single file** - All metadata in one place  
✅ **Better concurrency** - Multiple users/processes  
✅ **Easier backup** - Just copy `mlflow.db`  
✅ **Version control ready** - Add to git (or .gitignore)  

## File Structure

```
mlflow/
├── mlflow.db                  # SQLite database (experiment metadata)
├── mlartifacts/               # Experiment artifacts
│   ├── 0/                    # Experiment ID folders
│   ├── 1/
│   └── ...
├── mlflow_config.py          # Auto-configuration module
├── setup_mlflow_db.py        # Database setup script
└── .mlflow_config            # Environment config
```

## Querying the Database

You can query experiment data directly:

```python
import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect('mlflow.db')

# Query experiments
df = pd.read_sql_query("""
    SELECT name, experiment_id, lifecycle_stage
    FROM experiments
""", conn)

print(df)

# Query runs from an experiment
runs_df = pd.read_sql_query("""
    SELECT run_id, experiment_id, status, start_time, end_time
    FROM runs
    WHERE experiment_id = 0
""", conn)

conn.close()
```

## Backup and Restore

### Backup
```bash
# Copy the database file
copy mlflow\mlflow.db mlflow\mlflow_backup_2026-01-17.db

# Or use git (if not gitignored)
git add mlflow/mlflow.db
git commit -m "Backup MLflow experiments"
```

### Restore
```bash
# Replace with backup
copy mlflow\mlflow_backup_2026-01-17.db mlflow\mlflow.db
```

## Migrating from Old File-Based Storage

If you have existing runs in `mlruns/` directory:

```bash
# Keep old runs for reference
# They won't interfere with the new database

# Re-run important experiments to populate database
cd embedding_experiments
python run_experiments.py
```

## Troubleshooting

**Database locked error**:
- Close MLflow UI if running
- Ensure no other processes are accessing the database
- Check file permissions

**Can't see experiments in UI**:
- Verify database exists: `dir mlflow.db`
- Check tracking URI is set correctly
- Try: `mlflow ui --backend-store-uri sqlite:///mlflow.db`

**Artifacts not found**:
- Artifacts are in `mlartifacts/` directory
- They're stored separately from metadata
- Check experiment artifact location in database

## Advanced: PostgreSQL or MySQL

For production deployments, you can use PostgreSQL or MySQL:

```python
# PostgreSQL
tracking_uri = "postgresql://user:pass@localhost/mlflowdb"

# MySQL
tracking_uri = "mysql://user:pass@localhost/mlflowdb"

mlflow.set_tracking_uri(tracking_uri)
```

For thesis purposes, SQLite is recommended (simpler, no server needed).

## Git Considerations

### Option 1: Ignore database (recommended for large experiments)
Add to `.gitignore`:
```
mlflow/mlflow.db
mlflow/mlartifacts/
```

### Option 2: Include database (for reproducibility)
Include in git for complete reproducibility:
```bash
git add mlflow/mlflow.db
git add mlflow/mlartifacts/
```

**Note**: Database can grow large with many experiments. Consider selective backup.

---

## Quick Reference

| Task | Command |
|------|---------|
| Setup database | `python setup_mlflow_db.py` |
| Start UI | `mlflow ui --backend-store-uri sqlite:///mlflow.db` |
| Backup | `copy mlflow.db mlflow_backup.db` |
| Query | Connect with sqlite3 or pandas |
| Location | `mlflow/mlflow.db` |

✅ **Your experiments now use the database automatically!**
