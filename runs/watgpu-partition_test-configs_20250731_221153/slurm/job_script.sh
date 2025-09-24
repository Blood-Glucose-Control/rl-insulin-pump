#!/bin/bash
#SBATCH --job-name=rl-insulin-watgpu-partition_test-configs_20250731_221153
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=HI
#SBATCH --output=/u6/cjrisi/rl-insulin-pump/runs/watgpu-partition_test-configs_20250731_221153/slurm/slurm-%j.out
#SBATCH --error=/u6/cjrisi/rl-insulin-pump/runs/watgpu-partition_test-configs_20250731_221153/slurm/slurm-%j.err

# Print job information
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Run directory: /u6/cjrisi/rl-insulin-pump/runs/watgpu-partition_test-configs_20250731_221153"
echo "Configuration: /u6/cjrisi/rl-insulin-pump/runs/watgpu-partition_test-configs_20250731_221153/configs/watgpu-partition_test-configs.yaml"

# Change to project directory
cd /u6/cjrisi/rl-insulin-pump

# Activate virtual environment
echo "Activating virtual environment: env"
source env/bin/activate

# Save system information
echo "Python version: $(python --version)" > /u6/cjrisi/rl-insulin-pump/runs/watgpu-partition_test-configs_20250731_221153/logs/system_info.txt
echo "GPU information:" >> /u6/cjrisi/rl-insulin-pump/runs/watgpu-partition_test-configs_20250731_221153/logs/system_info.txt
nvidia-smi >> /u6/cjrisi/rl-insulin-pump/runs/watgpu-partition_test-configs_20250731_221153/logs/system_info.txt 2>/dev/null || echo "No GPU available" >> /u6/cjrisi/rl-insulin-pump/runs/watgpu-partition_test-configs_20250731_221153/logs/system_info.txt
echo "Environment packages:" >> /u6/cjrisi/rl-insulin-pump/runs/watgpu-partition_test-configs_20250731_221153/logs/system_info.txt
pip list >> /u6/cjrisi/rl-insulin-pump/runs/watgpu-partition_test-configs_20250731_221153/logs/system_info.txt

# Save git information if available
if [[ -d .git ]]; then
    echo "Git commit: $(git rev-parse HEAD)" >> /u6/cjrisi/rl-insulin-pump/runs/watgpu-partition_test-configs_20250731_221153/logs/system_info.txt
    echo "Git branch: $(git branch --show-current)" >> /u6/cjrisi/rl-insulin-pump/runs/watgpu-partition_test-configs_20250731_221153/logs/system_info.txt
    echo "Git status:" >> /u6/cjrisi/rl-insulin-pump/runs/watgpu-partition_test-configs_20250731_221153/logs/system_info.txt
    git status --porcelain >> /u6/cjrisi/rl-insulin-pump/runs/watgpu-partition_test-configs_20250731_221153/logs/system_info.txt
fi


# Run the training
echo "Starting training..."
python src/main.py --cfg "/u6/cjrisi/rl-insulin-pump/runs/watgpu-partition_test-configs_20250731_221153/configs/watgpu-partition_test-configs.yaml"

# Optionally run prediction after training
if [[ "false" == true ]]; then
    echo "Running prediction after training..."
    python src/main.py --cfg "/u6/cjrisi/rl-insulin-pump/runs/watgpu-partition_test-configs_20250731_221153/configs/watgpu-partition_test-configs.yaml" --mode predict
fi

# Save final model with run-specific name
if [[ -f "ddpg_simglucose.zip" ]]; then
    mv ddpg_simglucose.zip "/u6/cjrisi/rl-insulin-pump/runs/watgpu-partition_test-configs_20250731_221153/checkpoints/final_model.zip"
fi

# Move any other generated files to run directory
for file in *.pth *.txt _stable_baselines3_version; do
    if [[ -f "" ]]; then
        mv "" "/u6/cjrisi/rl-insulin-pump/runs/watgpu-partition_test-configs_20250731_221153/checkpoints/"
    fi
done

echo "Job completed at: $(date)"
echo "All outputs saved to: /u6/cjrisi/rl-insulin-pump/runs/watgpu-partition_test-configs_20250731_221153"
