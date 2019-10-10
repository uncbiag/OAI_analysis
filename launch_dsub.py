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
	if "GCP Auth Key Location" in line:
		temp1=line.split('GCP Auth Key Location=')
                gcp_key=temp1[1].strip()
	if "Month" in line:
		temp1=line.split('Month=')
		month=temp1[1].strip()
f.close()
'''
f=open(gcp_key,"r")
for line in f:
	gcp_private_key=gcp_private_key+line+'\n'
'''
dsub_temp="dsub --project " +project+" --zones "+zone+" --logging "+logging+" --input-recursive INPUT_PATH="+oai_data+" --output-recursive OUTPUT_PATH="+oai_results+" --machine-type "+machine_type+" --disk-size "+disk_size+" --accelerator-type "+gpu+" --accelerator-count 1 --image uray10/oai_image --provider google-v2 --env GCP_KEY="+gcp_key

for i in range(nnodes):
	if i==0:
		f=open("master_script.sh","w")
		f.write('gsutil cp '+gcp_key+' $HOME/.ssh\n')
		f.write('echo "OAI_data_sheet=/home/OAI_analysis/data" >> /home/OAI_analysis/config.txt\n')
		f.write('echo "OAI_data=$INPUT_PATH" >> /home/OAI_analysis/config.txt\n')
		f.write('echo "OAI_results=$OUTPUT_PATH" >> /home/OAI_analysis/config.txt\n')
		f.write('echo "Month='+month+'" >> /home/OAI_analysis/config.txt\n')
		#f.write("echo $GCP_KEY > $HOME/.ssh/gcp_key\n")
		f.write("mpirun -np "+str(nnodes)+" --hostfile /home/hosts python pipelines.py")
		f.close()
		dsub_temp=dsub_temp+" --script master_script.sh"
		os.system(dsub_temp+" > dsub_"+str(i)+".txt")
	else:
		f=open("script_"+str(i)+".sh","w")
		f.write('gsutil cp '+gcp_key+' $HOME/.ssh\n')
                f.write('echo "OAI_data_sheet=/home/OAI_analysis/data" >> /home/OAI_analysis/config.txt\n')
                f.write('echo "OAI_data=$INPUT_PATH" >> /home/OAI_analysis/config.txt\n')
                f.write('echo "OAI_results=$OUTPUT_PATH" >> /home/OAI_analysis/config.txt\n')
                f.write('echo "Month='+month+'" >> /home/OAI_analysis/config.txt')
		#f.write("echo $GCP_KEY > $HOME/.ssh/gcp_key")
		f.close()
		dsub_temp=dsub_temp+" --script script_"+str(i)+".sh"
		os.system(dsub_temp+" > dsub_"+str(i)+".txt")

for i in range(nnodes):
	f=open("dsub_"+str(i)+".txt","r")
	for line in f:
		job_name.append(line.strip('\n'))
	f.close()
	os.system("gcloud compute instances list --filter labels.job-id="+job_name[i]+" | awk 'NR>1 {print $4}' >> hostfile")
	os.system("gcloud compute instances list --filter labels.job-id="+job_name[i]+" | awk 'NR>1 {print $4}' > internal_ip.txt")
	f=open("internal_ip.txt","r")
	for line in f:
		internal_ip.append(line.strip('\n'))
	f.close()
	os.system("gcloud compute instances list --filter labels.job-id="+job_name[i]+" | awk 'NR>1 {print $5}' > external_ip.txt")
	for line in f:
                external_ip.append(line.strip('\n'))
	f.close()

for i in range(nnodes):
	os.system("scp -i "+gcp_key+" ./hostfile "+external_ip[i]+":~/.ssh/id_rsa")

