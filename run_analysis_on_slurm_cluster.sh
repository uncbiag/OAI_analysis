#!/bin/bash

# The name of the job:
#SBATCH --job-name="oai_run_analysis"

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=5-0:0:00

# Batch arrays
#SBATCH --array=0-199%15

# Send yourself an email when the job:
# aborts abnormally (fails)
#SBATCH --mail-type=FAIL
# begins
#SBATCH --mail-type=BEGIN
# ends successfully
#SBATCH --mail-type=END

# Use this email address:
#SBATCH --mail-user=mn@cs.unc.edu

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)

# The job command(s):
python mn_oai_pipeline.py --task_id ${SLURM_ARRAY_TASK_ID} --only_recompute_if_thickness_file_is_missing --knee_type LEFT_KNEE --progression_cohort_only
