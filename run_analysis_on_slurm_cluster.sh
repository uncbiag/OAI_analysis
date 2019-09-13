#!/bin/bash

# The name of the job:
#SBATCH --job-name="oai_run_analysis"

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-0:10:00

# compute partition
#SBATCH --qos gpu_access 
#SBATCH --partition gpu
#SBATCH --gres=gpu:1  

# Batch arrays
#--array=0-2105%15

#SBATCH --array=0-1%2

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

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# The job command(s):
source activate oai_mn

#python mn_oai_pipeline.py --task_id ${SLURM_ARRAY_TASK_ID} --output_directory /proj/mn/projects/oai/OAI_progression --config oai_analysis_longleaf.json --only_recompute_if_thickness_file_is_missing --knee_type LEFT_KNEE --progression_cohort_only

#python mn_oai_pipeline.py --task_id ${SLURM_ARRAY_TASK_ID} --output_directory /proj/mn/projects/oai/OAI_progression --config oai_analysis_longleaf.json --only_recompute_if_thickness_file_is_missing --knee_type BOTH_KNEES --get_number_of_jobs

python mn_oai_pipeline.py --task_id ${SLURM_ARRAY_TASK_ID} --config oai_analysis_longleaf.json --only_recompute_if_thickness_file_is_missing --knee_type BOTH_KNEES


