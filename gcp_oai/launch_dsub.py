import os
dsub_temp=""
job_name=[]
gcp_private_key=""
internal_ip=[]
external_ip=[]

f=open("./config.txt","r")
for line in f:
	if "Project" in line:
		temp1=line.split('Project=')
		project=temp1[1].strip()
	if "OAI_data" in line:
		temp1=line.split('OAI_data=')
		oai_data=temp1[1].strip()
	if "OAI_results" in line:
		temp1=line.split('OAI_results=')
		oai_results=temp1[1].strip()
	if "GPU" in line:
		temp1=line.split('GPU=')
		gpu=temp1[1].strip()
	if "Machine Type" in line:
		temp1=line.split('Machine Type=')
		machine_type=temp1[1].strip()
	if "Logging" in line:
		temp1=line.split('Logging=')
		logging=temp1[1].strip()
	if "Disk Size" in line:
		temp1=line.split('Disk Size=')
		disk_size=temp1[1].strip()
	if "NNodes" in line:
		temp1=line.split('NNodes=')
		nnodes=temp1[1].strip()
		nnodes=int(nnodes)
	if "Zone" in line:
		temp1=line.split('Zone=')
		zone=temp1[1].strip()
		zone='"'+zone+'"'
	if "Month" in line:
		temp1=line.split('Month=')
		month=temp1[1].strip()
f.close()

dsub_temp="dsub --project " +project+" --zones "+zone+" --logging "+logging+" --output-recursive OUTPUT_PATH="+oai_results+" --script script_n.sh --min-ram 30 --min-cores 8 --disk-size 500 --accelerator-type nvidia-tesla-k80 --accelerator-count 1 --image uray10/oai:dsub_update_2 --provider google-v2"

for i in range(nnodes):
	f=open("script_"+str(i)+".sh","w")
	f.write("#!/bin/bash\n")
	f.write("python3.6 /home/OAI_analysis/execute.py --month "+month+" --OAI_data "+oai_data+" --cloud google --size "+str(nnodes)+" --rank "+str(i)+" --OAI_data_sheet /home/OAI_analysis/tmp.csv --OAI_results ${OUTPUT_PATH}\n")
	#f.write("python /home/OAI_analysis/parallel_pipeline.py --month "+month+" --OAI_data_sheet /home/OAI_analysis/SEG_3D_DESS_6visits.csv --size "+str(nnodes)+" --rank "+str(i)+" --OAI_results ${OUTPUT_PATH}")
	f.close()
	dsub_tempo=dsub_temp.replace("script_n.sh", "script_"+str(i)+".sh")
	#os.system(dsub_tempo+" > dsub_"+str(i)+".txt")
	print(dsub_tempo)
