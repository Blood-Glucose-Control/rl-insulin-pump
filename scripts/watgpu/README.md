# RL Insulin Pump Training Management

This directory contains improved scripts for managing training runs in a more organized way.

## Scripts

### 1. `submit_training_job.sh` - Enhanced Training Submission
**Purpose**: Submit SLURM jobs with organized output directories

**Usage**:
```bash
# Basic usage with required config
./submit_training_job.sh --config ../configs/watgpu-partition_test-configs.yaml

# Full usage with all options
./submit_training_job.sh \
    --config ../configs/watgpu-partition_test-configs.yaml \
    --run-name "ddpg_experiment_v1" \
    --time "02:00:00" \
    --mem "16G" \
    --cpus 8 \
    --gres "gpu:2" \
    --partition "HI" \
    --email "your.email@domain.com"
```

**Features**:
- Creates self-contained run directories under `runs/`
- Automatically modifies config to use run-specific paths
- Organizes all outputs (logs, checkpoints, tensorboard, results)
- Includes system info and git state
- Provides monitoring commands

### 2. `organize_previous_run.sh` - Retroactive Organization
**Purpose**: Organize files from previous unstructured runs

**Usage**:
```bash
./organize_previous_run.sh
```

**What it does**:
- Collects scattered files from your previous run (slurm-1335771.out)
- Organizes them into a structured directory
- Creates analysis scripts
- Generates plots and summaries

### 3. `batch_submit.sh` - Original Script (Legacy)
The original submission script - kept for reference but recommend using the new enhanced version.

## Directory Structure

After using the new scripts, your runs will be organized like this:

```
runs/
├── ddpg_experiment_v1_20250715_143022/
│   ├── configs/
│   │   └── watgpu-partition_test-configs.yaml
│   ├── checkpoints/
│   │   ├── final_model.zip
│   │   ├── ddpg_checkpoint_*.zip
│   │   └── *.pth files
│   ├── tensorboard/
│   │   └── DDPG_*/
│   ├── logs/
│   │   ├── system_info.txt
│   │   └── training logs
│   ├── results/
│   │   ├── training_metrics.csv
│   │   └── analysis plots
│   ├── slurm/
│   │   ├── slurm-*.out
│   │   ├── slurm-*.err
│   │   └── job_script.sh
│   ├── run_summary.txt
│   └── analyze_run.py
```

## Reconstructing the Original Command

Based on the slurm-1335771.out file, the original command was likely:

```bash
# What probably ran originally (reconstructed):
sbatch << 'EOF'
#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=HI
#SBATCH --output=slurm-%j.out

cd /u6/cjrisi/rl-insulin-pump
source env/bin/activate
python src/main.py --cfg configs/watgpu-partition_test-configs.yaml
EOF
```

## Quick Start for New Training

1. **For a new organized training run**:
   ```bash
   cd /u6/cjrisi/rl-insulin-pump/scripts/watgpu
   ./submit_training_job.sh --config ../../configs/watgpu-partition_test-configs.yaml --run-name "my_experiment"
   ```

2. **To organize your previous run**:
   ```bash
   cd /u6/cjrisi/rl-insulin-pump/scripts/watgpu
   ./organize_previous_run.sh
   ```

3. **To monitor a running job**:
   ```bash
   squeue -u $(whoami)
   tail -f runs/your_run_name/slurm/slurm-*.out
   ```

4. **To view TensorBoard**:
   ```bash
   tensorboard --logdir runs/your_run_name/tensorboard/
   ```

## Benefits of the New System

1. **Self-contained runs**: Each experiment has its own directory
2. **No file conflicts**: Multiple experiments can run simultaneously
3. **Easy comparison**: All runs organized under `runs/`
4. **Reproducibility**: Config and system info saved with each run
5. **Analysis tools**: Automatic generation of analysis scripts
6. **Clean workspace**: No scattered files in the root directory

## Migration from Old System

If you have existing scattered files like the ones that created slurm-1335771.out:

1. Run `./organize_previous_run.sh` to collect them
2. Use the new `./submit_training_job.sh` for future runs
3. Keep the old files as backup until you're confident in the new system
