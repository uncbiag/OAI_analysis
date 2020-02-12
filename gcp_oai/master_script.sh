gsutil cp gs://gbsc-gcp-project-annohive-dev-user-uray/input_oai/google_compute_engine $HOME/.ssh
echo "OAI_data_sheet=/home/OAI_analysis/data" >> /home/OAI_analysis/config.txt
echo "OAI_data=$INPUT_PATH" >> /home/OAI_analysis/config.txt
echo "OAI_results=$OUTPUT_PATH" >> /home/OAI_analysis/config.txt
echo "Month=12" >> /home/OAI_analysis/config.txt
mpirun -np 3 --hostfile /home/hosts python pipelines.py