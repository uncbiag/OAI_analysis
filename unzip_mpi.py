import os
from mpi4py import MPI

file_name="36m.zip"
file_name_small="36m"
folder_1="5.C.1"
folder_2="5.E.1"
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


comm=MPI.COMM_WORLD
os.system('mkdir /tmp/uray')
os.system('mkdir /tmp/uray/store')
os.system('mkdir /tmp/uray/store/'+folder_1)
os.system('mkdir /tmp/uray/store/'+folder_2)


folder_1_partition = int((len(list_folder_1))/comm.size)
folder_2_partition = int((len(list_folder_2))/comm.size)

folder_1_remaining = ((len(list_folder_1))%comm.size)
folder_2_remaining = ((len(list_folder_2))%comm.size)

stride_1 = comm.rank*folder_1_partition
stride_2 = comm.rank*folder_2_partition

f_tmp_csv=open("/tmp/uray/SEG_3D_DESS_6visits.csv","w")
with open('parsed_csv.csv') as f_sample_csv:
        first_line = f_sample_csv.readline()
        f_tmp_csv.write(first_line)
f_tmp_csv.close()

set_k=set()
for k in range(stride_1,((comm.rank*folder_1_partition)+folder_1_partition)):
	os.system("mkdir /tmp/uray/store/"+folder_1+"/"+list_folder_1[k])
	f_first=open("/tmp/uray/First.txt","a")
	f_parsed_csv=open("./parsed_csv.csv","r")
	f_tmp_csv=open("/tmp/uray/SEG_3D_DESS_6visits.csv","a")
	for line in f_parsed_csv:
		if list_folder_1[k] in line and folder_1 in line:
			f_tmp_csv.write(line)
			set_k.add(list_folder_1[k])
			if(list_folder_1_flag[k] is "no"):
				f_first.write(list_folder_1[k]+"\n")
				list_folder_1_flag[k]="yes"
	f_parsed_csv.close()
	f_tmp_csv.close()
	f_first.close()

set_m=set()
if folder_1_remaining > 0:
	for m in range((((comm.size-1)*folder_1_partition)+folder_1_partition),len(list_folder_1)):
		if ((len(list_folder_1))%m)==comm.rank:
			os.system("mkdir /tmp/uray/store/"+folder_1+"/"+list_folder_1[m])
			f_first=open("/tmp/uray/First.txt","a")
			f_parsed_csv=open("./parsed_csv.csv","r")
			f_tmp_csv=open("/tmp/uray/SEG_3D_DESS_6visits.csv","a")
			for line in f_parsed_csv:
				if list_folder_1[m] in line and folder_1 in line:
					f_tmp_csv.write(line)
					set_m.add(list_folder_1[m])
					if(list_folder_1_flag[m] is "no"):
						f_first.write(list_folder_1[m]+"\n")
						list_folder_1_flag[m]="yes"
			f_parsed_csv.close()
			f_tmp_csv.close()
			f_first.close()

set_l=set()
for l in range(stride_2,((comm.rank*folder_2_partition)+folder_2_partition)):
	os.system("mkdir /tmp/uray/store/"+folder_2+"/"+list_folder_2[l])
	f_second=open("/tmp/uray/Second.txt","a")
	f_parsed_csv=open("./parsed_csv.csv","r")
	f_tmp_csv=open("/tmp/uray/SEG_3D_DESS_6visits.csv","a")
	for line in f_parsed_csv:
		if list_folder_2[l] in line and folder_2 in line:
			f_tmp_csv.write(line)
			set_l.add(list_folder_2[l])
			if(list_folder_2_flag[l] is "no"):
				f_second.write(list_folder_2[l]+"\n")
				list_folder_2_flag[l]="yes"
	f_parsed_csv.close()
	f_tmp_csv.close()
	f_second.close()

set_n=set()
if folder_2_remaining > 0:
	for n in range((((comm.size-1)*folder_2_partition)+folder_2_partition),len(list_folder_2)):
		if ((len(list_folder_2))%n)==comm.rank:
			os.system("mkdir /tmp/uray/store/"+folder_2+"/"+list_folder_2[n])
			f_second=open("/tmp/uray/Second.txt","a")
			f_parsed_csv=open("./parsed_csv.csv","r")
			f_tmp_csv=open("/tmp/uray/SEG_3D_DESS_6visits.csv","a")
			for line in f_parsed_csv:
				if list_folder_2[n] in line and folder_2 in line:
					f_tmp_csv.write(line)
					set_n.add(list_folder_2[n])
					if(list_folder_2_flag[n] is "no"):
						f_second.write(list_folder_2[n]+"\n")
						list_folder_2_flag[n]="yes"
			f_parsed_csv.close()
			f_tmp_csv.close()
			f_second.close()

comm.Barrier()

list_folder_1_1=[]
for k in range(stride_1,((comm.rank*folder_1_partition)+folder_1_partition)):
	f_tmp_csv=open("/tmp/uray/SEG_3D_DESS_6visits.csv","r")
	#f_tmp_csv=open("/tmp/uray/First.txt","r")
	for line in f_tmp_csv:
		if list_folder_1[k] in line:
			if list_folder_1[k] not in list_folder_1_1:
		#if list_folder_1[k] in set_k:
		#if os.path.exists('/tmp/uray/'+folder_1+'/'+list_folder_1[k]) is False:
				os.system("unzip "+file_name+" '"+folder_1+"/"+list_folder_1[k]+"/*' -d /tmp/uray/")
			#os.system("mv -v -f /tmp/uray/store/"+folder_1+"/"+list_folder_1[k]+"/"+folder_1+"/"+list_folder_1[k]+" /tmp/uray/"+folder_1)
			#set_k.remove(list_folder_1[k])
				list_folder_1_1.append(list_folder_1[k])
			#os.system('unzip '+file_name+' '+folder_1+'/'+list_folder_1[k]+'/* -d '+'/tmp/uray/')
			#os.system('unzip -o '+file_name+' '+folder_1+'/'+list_folder_1[k]+'/\*')
			#os.system('cd '+folder_1+' && '+'mv '+list_folder_1[k]+' /tmp/uray/'+folder_1+' && cd ..')
			#os.system('cd '+folder_1+' && '+'cp -r ./'+list_folder_1[k]+' /tmp/uray/'+folder_1+' && rm -r '+list_folder_1[k]+' && cd ..')
			#os.system("mkdir /tmp/uray/store/"+folder_1+"/"+list_folder_1[k])
			#os.system("rsync -av /tmp/uray/store/"+folder_1+"/"+list_folder_1[k]+"/"+folder_1+"/"+list_folder_1[k]+" /tmp/uray/"+folder_1)
			#os.system("rm -rf /tmp/uray/store/"+folder_1+"/"+list_folder_1[k]+"/*")
			#print('PROCESSOR: '+MPI.Get_processor_name()+' unzip -o '+file_name+' '+folder_1+'/'+list_folder_1[k]+'/\*')
			#print('PROCESSOR: '+MPI.Get_processor_name()+' cd '+folder_1+' && '+'cp -r ./'+list_folder_1[k]+' /tmp/uray/'+folder_1+' && rm -r '+list_folder_1[k]+' && cd ..')
	f_tmp_csv.close()
comm.Barrier()

list_folder_1_2=[]
if folder_1_remaining > 0:
	for m in range((((comm.size-1)*folder_1_partition)+folder_1_partition),len(list_folder_1)):
		if ((len(list_folder_1))%m)==comm.rank:
			f_tmp_csv=open("/tmp/uray/SEG_3D_DESS_6visits.csv","r")
			#f_tmp_csv=open("/tmp/uray/First.txt","r")
			for line in f_tmp_csv:
				if list_folder_1[m] in line:
					#if list_folder_1[m] in set_m:
					if list_folder_1[m] not in list_folder_1_2:
					#if os.path.exists('/tmp/uray/'+folder_1+'/'+list_folder_1[m]) is False:
						os.system("unzip "+file_name+" '"+folder_1+"/"+list_folder_1[m]+"/*' -d /tmp/uray/")
						#os.system("mv -v -f /tmp/uray/store/"+folder_1+"/"+list_folder_1[m]+"/"+folder_1+"/"+list_folder_1[m]+" /tmp/uray/"+folder_1)
						#set_m.remove(list_folder_1[m])
						list_folder_1_2.append(list_folder_1[m])
					#print('PROCESSOR: '+MPI.Get_processor_name()+' unzip -o '+file_name+' '+folder_1+'/'+list_folder_1[m]+'/\*')
					#print('PROCESSOR: '+MPI.Get_processor_name()+' cd '+folder_1+' && '+'cp -r ./'+list_folder_1[k]+' /tmp/uray/'+folder_1+' && rm -r '+list_folder_1[k]+' && cd ..')
					#os.system('unzip '+file_name+' '+folder_1+'/'+list_folder_1[m]+'/* -d '+'/tmp/uray/')
					#os.system('unzip -o '+file_name+' '+folder_1+'/'+list_folder_1[m]+'/\*')
					#os.system('cd '+folder_1+' && '+'mv '+list_folder_1[m]+' /tmp/uray/'+folder_1+' && cd ..')
					#os.system('cd '+folder_1+' && '+'cp -r ./'+list_folder_1[m]+' /tmp/uray/'+folder_1+' && rm -r '+list_folder_1[m]+' && cd ..')
					#os.system("mkdir /tmp/uray/store/"+folder_1+"/"+list_folder_1[m])
					#os.system("rsync -av /tmp/uray/store/"+folder_1+"/"+list_folder_1[m]+"/"+folder_1+"/"+list_folder_1[m]+" /tmp/uray/"+folder_1)
					#os.system("rm -rf /tmp/uray/store/"+folder_1+"/"+list_folder_1[m]+"/*")
			f_tmp_csv.close()
comm.Barrier()

list_folder_2_1=[]
for l in range(stride_2,((comm.rank*folder_2_partition)+folder_2_partition)):
	f_tmp_csv=open("/tmp/uray/SEG_3D_DESS_6visits.csv","r")
	#f_tmp_csv=open("/tmp/uray/Second.txt","r")
	for line in f_tmp_csv:
		if list_folder_2[l] in line:
			#if list_folder_2[l] in set_l:
			if list_folder_2[l] not in list_folder_2_1:
			#if os.path.exists('/tmp/uray/'+folder_2+'/'+list_folder_2[l]) is False:
				os.system("unzip "+file_name+" '"+folder_2+"/"+list_folder_2[l]+"/*' -d /tmp/uray/")
				#os.system("mv -v -f /tmp/uray/store/"+folder_2+"/"+list_folder_2[l]+"/"+folder_2+"/"+list_folder_2[l]+" /tmp/uray/"+folder_2)
				#set_l.remove(list_folder_2[l])
				list_folder_2_1.append(list_folder_2[l])
			#os.system('unzip '+file_name+' '+folder_2+'/'+list_folder_2[l]+'/* -d '+'/tmp/uray/')
			#os.system('unzip -o '+file_name+' '+folder_2+'/'+list_folder_2[l]+'/\*')
			#os.system('cd '+folder_2+' && '+'cp -r ./'+list_folder_2[l]+' /tmp/uray/'+folder_2+' && rm -r '+list_folder_2[l]+' && cd ..')
			#print('PROCESSOR: '+MPI.Get_processor_name()+' unzip -o '+file_name+' '+folder_2+'/'+list_folder_2[l]+'/\*')
			#print('PROCESSOR: '+MPI.Get_processor_name()+' cd '+folder_2+' && '+'cp -r ./'+list_folder_2[l]+' /tmp/uray/'+folder_2+' && rm -r '+list_folder_2[l]+' && cd ..')
			#os.system("mkdir /tmp/uray/store/"+folder_2+"/"+list_folder_2[l])
			#os.system("rsync -av /tmp/uray/store/"+folder_2+"/"+list_folder_2[l]+"/"+folder_2+"/"+list_folder_2[l]+" /tmp/uray/"+folder_2)
			#os.system("rm -rf /tmp/uray/store/"+folder_2+"/"+list_folder_2[l]+"/*")
	f_tmp_csv.close()
comm.Barrier()

list_folder_2_2=[]
if folder_2_remaining > 0:
	for n in range((((comm.size-1)*folder_2_partition)+folder_2_partition),len(list_folder_2)):
		if ((len(list_folder_2))%n)==comm.rank:
			f_tmp_csv=open("/tmp/uray/SEG_3D_DESS_6visits.csv","r")
			#f_tmp_csv=open("/tmp/uray/Second.txt","r")
			for line in f_tmp_csv:
				if list_folder_2[n] in line:
					#if list_folder_2[n] in set_n:
					if list_folder_2[n] not in list_folder_2_2:
					#if os.path.exists('/tmp/uray/'+folder_2+'/'+list_folder_2[n]) is False:
						os.system("unzip "+file_name+" '"+folder_2+"/"+list_folder_2[n]+"/*' -d /tmp/uray/")
						#os.system("mv -v -f /tmp/uray/store/"+folder_2+"/"+list_folder_2[n]+"/"+folder_2+"/"+list_folder_2[n]+" /tmp/uray/"+folder_2)
						#set_n.remove(list_folder_2[n])
						list_folder_2_2.append(list_folder_2[n])
					#print('PROCESSOR: '+MPI.Get_processor_name()+' unzip -o '+file_name+' '+folder_2+'/'+list_folder_2[n]+'/\*')
					#print('PROCESSOR: '+MPI.Get_processor_name()+' cd '+folder_2+' && '+'cp -r ./'+list_folder_2[n]+' /tmp/uray/'+folder_2+' && rm -r '+list_folder_2[n]+' && cd ..')
					#os.system('unzip '+file_name+' '+folder_2+'/'+list_folder_2[n]+'/* -d '+'/tmp/uray/')
					#os.system('unzip -o '+file_name+' '+folder_2+'/'+list_folder_2[n]+'/\*')
					#os.system('cd '+folder_2+' && '+'cp -r ./'+list_folder_2[n]+' /tmp/uray/'+folder_2+' && rm -r '+list_folder_2[n]+' && cd ..')
					#os.system("mkdir /tmp/uray/store/"+folder_2+"/"+list_folder_2[n])
					#os.system("rsync -av /tmp/uray/store/"+folder_2+"/"+list_folder_2[n]+"/"+folder_2+"/"+list_folder_2[n]+" /tmp/uray/"+folder_2)
					#os.system("rm -rf /tmp/uray/store/"+folder_2+"/"+list_folder_2[n]+"/*")
			f_tmp_csv.close()
comm.Barrier()
