f=open("./config.txt","r")
for line in f:
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

if cloud == "google":
	for i in range(nnodes):
		f=open("script_"+str(i)+".sh","w")
		f.write("docker run -td uray10/oai:secondversion\n")
		f.write("docker exec $(docker ps -l | awk 'NR>1 {print $10}') python /home/OAI_analysis/execute.py --month "+month+" --oai_data "+oai_data+" --cloud "+cloud+"\n")
		if i==0:
			f.write("docker exec $(docker ps -l | awk 'NR>1 {print $10}') mpirun -np "+str(nnodes)+" --hostfile /home/hosts python /home/OAI_analysis/execute_pipeline.py --month "+month+" --oai_data "+oai_data+" --cloud "+cloud+" --oai_results "+oai_results+" --OAI_data_sheet /home/OAI_analysis/data/SEG_3D_DESS_6visits.csv")
		f.close()

		os.system("gcloud compute instances create-with-container instancce-oai-"+str(i)+" --zone "+zone+" --container-image=uray10/oai:firstversion --machine-type="+machine_type+" --boot-disk-size="+disk_size+" --accelerator="+gpu)

		os.system("gcloud compute instances list --filter name=instance-oai-"+str(i)+" | awk 'NR>1 {print $4}' >> hosts")
		os.system("gcloud compute instances list --filter name=instance-oai-"+str(i)+" | awk 'NR>1 {print $4}' > internal_ip.txt")
		f=open("internal_ip.txt","r")
		for line in f:
			internal_ip.append(line.strip('\n'))
		f.close()
		os.system("gcloud compute instances list --filter labels.job-id="+job_name[i]+" | awk 'NR>1 {print $5}' > external_ip.txt")
		for line in f:
			external_ip.append(line.strip('\n'))
		f.close()

		os.system("scp -i "+gcp_key+" "+gcp_key+" "+external_ip[i]+":~/.ssh/id_rsa"

	for i in range(nnodes):
		os.system("scp -i "+gcp_key+" ./hosts "+external_ip[i]+":/home/hosts")
