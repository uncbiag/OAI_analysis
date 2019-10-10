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

#os.system("gsutil cp "+oai_data_sheet+" .")
oai_data_sheet="SEG_3D_DESS_6visits.csv"

first_folder_names_list=["1.C.2","3.C.2","5.C.1","6.C.1","8.C.1"]
second_folder_names_list=["1.E.1","3.E.1""5.E.1","6.E.1","8.E.1"]
months_list=["12","24","36","48","72"]
file_name_list=["12m.zip","24m.zip","36m.zip","48m.zip","72m.zip"]
file_name_small_list=["12m","24m","36m","48m","72m"]
month_index=months_list.index(month)
OAI_month=int(month)

first_folder=first_folder_names_list[month_index]
second_folder=second_folder_names_list[month_index]

first_set=set()
second_set=set()

first_list=[]
second_list=[]

f_parsed_csv=open("./parsed_csv.csv","w")
with open("./SEG_3D_DESS_6visits.csv") as f_original_csv:
	first_line = f_original_csv.readline()
        f_parsed_csv.write(first_line)
f_parsed_csv.close()

f_original_csv=open("./SEG_3D_DESS_6visits.csv","r")
f_parsed_csv=open("./parsed_csv.csv","a")
for line in f_original_csv:
        if first_folder in line:
		f_parsed_csv.write(line)
		split_1=line.split(first_folder+"/")
		split_1_1=split_1[1].split("/")
		first_set.add(split_1_1[0])
        if second_folder in line:
		f_parsed_csv.write(line)
		split_2=line.split(second_folder+"/")
		split_2_1=split_2[1].split("/")
		second_set.add(split_2_1[0])
f_original_csv.close()
f_parsed_csv.close()

first_list=list(first_set)
second_list=list(second_set)

f_first_list=open("./First_Folder_List.txt","w")
for i in range(len(first_list)):
        f_first_list.write(first_list[i]+"\n")
f_first_list.close()

f_second_list=open("./Second_Folder_List.txt","w")
for i in range(len(second_list)):
        f_second_list.write(second_list[i]+"\n")
f_second_list.close()

file_name=file_name_list[month_index]
file_name_small=file_name_small_list[month_index]

folder_1=first_folder_names_list[month_index]
folder_2=second_folder_names_list[month_index]

list_folder_1=[]
list_folder_2=[]
list_folder_1_flag=[]
list_folder_2_flag=[]
set1=()
set2=()
f_first=open("./First_Folder_List.txt","r")
for line in f_first:
        line=line.rstrip('\n')
        list_folder_1.append(line)
        list_folder_1_flag.append("no")
f_first.close()

f_second=open("./Second_Folder_List.txt","r")
for line in f_second:
        line=line.rstrip('\n')
        list_folder_2.append(line)
        list_folder_2_flag.append("no")
f_second.close()

folder_1_partition = int((len(list_folder_1))/nnodes)
folder_2_partition = int((len(list_folder_2))/nnodes)

folder_1_remaining = ((len(list_folder_1))%nnodes)
folder_2_remaining = ((len(list_folder_2))%nnodes)

for i in range(nnodes):
	stride_1 = i*folder_1_partition
	stride_2 = i*folder_2_partition
	f_tmp_csv=open(oai_data_sheet+"."+str(i),"w")
	with open('parsed_csv.csv') as f_sample_csv:
		first_line = f_sample_csv.readline()
		f_tmp_csv.write(first_line)
	f_tmp_csv.close()
	set_k=set()
	for k in range(stride_1,((i*folder_1_partition)+folder_1_partition)):
		f_parsed_csv=open("./parsed_csv.csv","r")
		f_tmp_csv=open(oai_data_sheet+"."+str(i),"a")
		for line in f_parsed_csv:
			if list_folder_1[k] in line and folder_1 in line:
				f_tmp_csv.write(line)
				set_k.add(list_folder_1[k])
				if(list_folder_1_flag[k] is "no"):
			    		list_folder_1_flag[k]="yes"
		f_parsed_csv.close()
		f_tmp_csv.close()

	set_m=set()
	if folder_1_remaining > 0:
		for m in range((((nnodes-1)*folder_1_partition)+folder_1_partition),len(list_folder_1)):
			if ((len(list_folder_1))%m)==i:
				f_parsed_csv=open("./parsed_csv.csv","r")
				f_tmp_csv=open(oai_data_sheet+"."+str(i),"a")
				for line in f_parsed_csv:
			    		if list_folder_1[m] in line and folder_1 in line:
						f_tmp_csv.write(line)
						set_m.add(list_folder_1[m])
						if(list_folder_1_flag[m] is "no"):
				    			list_folder_1_flag[m]="yes"
				f_parsed_csv.close()
				f_tmp_csv.close()

	set_l=set()
	for l in range(stride_2,((i*folder_2_partition)+folder_2_partition)):
		f_parsed_csv=open("./parsed_csv.csv","r")
		f_tmp_csv=open(oai_data_sheet+"."+str(i),"a")
		for line in f_parsed_csv:
			if list_folder_2[l] in line and folder_2 in line:
				f_tmp_csv.write(line)
				set_l.add(list_folder_2[l])
				if(list_folder_2_flag[l] is "no"):
					list_folder_2_flag[l]="yes"
		f_parsed_csv.close()
		f_tmp_csv.close()

	set_n=set()
	if folder_2_remaining > 0:
		for n in range((((nnodes-1)*folder_2_partition)+folder_2_partition),len(list_folder_2)):
		    	if ((len(list_folder_2))%n)==i:
				f_parsed_csv=open("./parsed_csv.csv","r")
				f_tmp_csv=open(oai_data_sheet+"."+str(i),"a")
				for line in f_parsed_csv:
			    		if list_folder_2[n] in line and folder_2 in line:
						f_tmp_csv.write(line)
						set_n.add(list_folder_2[n])
						if(list_folder_2_flag[n] is "no"):
				    			f_second.write(list_folder_2[n]+"\n")
				    			list_folder_2_flag[n]="yes"
				f_parsed_csv.close()
				f_tmp_csv.close()


dsub_temp="dsub --project " +project+" --zones "+zone+" --logging "+logging+" --input-recursive INPUT_PATH="+oai_data+" --output-recursive OUTPUT_PATH="+oai_results+" --machine-type "+machine_type+" --disk-size "+disk_size+" --accelerator-type "+gpu+" --accelerator-count 1 --image uray10/oai_image --provider google-v2"

for i in range(nnodes):
	dsub_temp=dsub_temp+" --env RANK="+str(i)+" --script script_"+str(i)+".sh"
	os.system("gsutil cp "+oai_data_sheet+"."+str(i)+" "+oai_data)
	f=open("script_"+str(i)+".sh","w")
	f.write('echo "OAI_data_sheet=${INPUT_PATH}" >> /home/OAI_analysis/config.txt\n')
	f.write('echo "OAI_data=${INPUT_PATH}" >> /home/OAI_analysis/config.txt\n')
	f.write('echo "OAI_results=${OUTPUT_PATH}" >> /home/OAI_analysis/config.txt\n')
	f.write('echo "Month='+month+'" >> /home/OAI_analysis/config.txt\n')
	f.write('gsutil cp '+oai_data+'/'+oai_data_sheet+"."+str(i)+' /home/OAI_analysis/\n')
	f.write('gsutil cp gs://oai_bucket/pipelines_without_mpi.py /home/OAI_analysis/\n')
	f.write('python /home/OAI_analysis/pipelines_without_mpi.py\n')
	f.close()
	os.system(dsub_temp)
