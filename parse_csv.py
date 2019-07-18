import os

first_folder="5.C.1"
second_folder="5.E.1"

first_set=set()
second_set=set()

first_list=[]
second_list=[]

f_parsed_csv=open("./parsed_csv.csv","w")
with open("/home/uray/temp_OAI/OAI_analysis-master/data/SEG_3D_DESS_6visits.csv") as f_original_csv:
	first_line = f_original_csv.readline()
	f_parsed_csv.write(first_line)
f_parsed_csv.close()

f_original_csv=open("/home/uray/temp_OAI/OAI_analysis-master/data/SEG_3D_DESS_6visits.csv","r")
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
