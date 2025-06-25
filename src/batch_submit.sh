#!/bin/bash

# To be submitted to the SLURM queue with the command:
# sbatch batch-submit.sh

# Initialize variables
OUT=""
ERR=""
EMAIL=""
TIME=""
MEM=""
ENV=""
CPUS=""
GRES=""
USRGRP=""
CMD=()

while [[ "$#" -gt 0 ]]; do
   case "$1" in
      '--out') OUT="$2"; shift 2;;
      '--err')  ERR="$2"; shift 2 ;;
      '--email') EMAIL="$2"; shift 2 ;;
      '--time') TIME="$2"; shift 2 ;;
      '--mem') MEM="$2"; shift 2 ;;
      '--env') ENV="$2"; shift 2 ;;
      '--cpus') CPUS="$2"; shift 2 ;;
      '--gres') GRES="$2"; shift 2 ;;
      '--partition') USRGRP="$2"; shift 2 ;;
      *) CMD+=("$1"); shift ;;
   esac
done

# Debugging: Print all parsed variables (optional)
echo "OUT: $OUT"
echo "ERR: $ERR"
echo "EMAIL: $EMAIL"
echo "TIME: $TIME"
echo "MEM: $MEM"
echo "ENV_VARS: $ENV_VARS"
echo "CPUS: $CPUS"
echo "GRES: $GRES"
echo "USRGRP: $USRGRP"
echo "COMMAND: ${CMD[*]}"

# Ensure a command was provided
if [[ ${#CMD[@]} -eq 0 ]]; then
    echo "Error: No command provided."
    exit 1
fi

# Set resource requirements: Queues are limited to seven day allocations
# Time format: HH:MM:SS
#SBATCH --time="$TIME"
#SBATCH --mem="$MEM"
#SBATCH --cpus-per-task="$CPUS"
#SBATCH --gres="$GRES"
#SBATCH --partition="$USRGRP"

# Set output file destinations (optional)
# By default, output will appear in a file in the submission directory:
# slurm-$job_number.out
# This can be changed:
if [[ -n "$OUT" ]]; then
   echo "setting output file to $OUT"
   #SBATCH -o "$OUT" # File to which STDOUT will be written
fi
if [[ -n "$ERR" ]]; then
   echo "setting error file to $ERR"
   #SBATCH -e "$ERR" # File to which STDERR will be written
fi

if [[ -n "$EMAIL" ]]; then
   # email notifications: Get email when your job starts, stops, fails, completes...
   # Set email address
   echo "setting notification email address to $EMAIL"
   #SBATCH --mail-user="$EMAIL"
   # Set types of notifications (from the options: BEGIN, END, FAIL, REQUEUE, ALL):
   #SBATCH --mail-type=ALL
fi

if [[ -n "$ENV" ]]; then
   echo "loading conda environment $ENV"
   # Load up your conda environment
   # Set up environment on watgpu.cs or in interactive session (use `source` keyword instead of `conda`)
   source activate "$ENV"
fi

# Task to run

${CMD[@]}
