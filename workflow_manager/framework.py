f=open("./config.txt","r")
for line in f:
    #parse environment first and accordingly call function
    if "Environment" in line:
        temp1=line.split('Environment=')
        environ=temp1[1].strip()

    all_environ = {"local" : local,
                   "gcp"   : google,
                   "aws"   : aws
    }
    all_environ[environ]()

def local():
    command=[]
    #parse config file
    f=open("./config.txt","r")
    for line in f:
        if "Docker Image" in line:
            temp1=line.split('Docker Image=')
            d_image=temp1[1].strip()
        if "Command" in line:
            temp1=line.split('Command=')
            command.append(temp1[1].strip())
        if "Input Path" in line:
            temp1=line.split('Input Path=')
            input_path=temp1[1].strip()
        if "Output Path" in line:
            temp1=line.split('Output Path=')
            output_path=temp1[1].strip()
    f.close()
    
    #run docker image in background
    os.system("sudo docker run -dit "+d_image)
    #get name of container
    os.system('sudo docker ps --format "{{.Names}} > d_name.txt')
    
    f=open("./d_name.txt","r")
    for line in f:
        d_name=line.strip()
    f.close()

    for i in range(len(command):
        #create folder in container to store input file
        os.system("sudo docker exec "+d_name+" mkdir /input_"+str(i))
        #create  folder in container to store output file
        os.system("sudo docker exec "+d_name+" mkdir /output_"+str(i))
        #in command user will specify input and output path as ${INPUT_PATH} and ${OUTPUT_PATH} 
        command[i].replace("${INPUT_PATH}","/input_"+str(i))
        command[i].replace("${OUTPUT_PATH}","/output_"+str(i))

    #copy files from input path to input path in container
    os.system("sudo docker cp "+input_path+" "+d_name+":/input_1")
    
    #execute command
    for i in range(len(command):
        os.system("sudo docker exec "+d_name+" "+command[i])

    #copy files from inside container to output path outside container
    os.system("sudo docker cp "+d_name+":/output_"+str(len(command)-1)+" "+output_path)

def gcp():

    #number of virtual cpus of each instance in google cloud
    vcpu=[2,4,8,16,32,2,4,8,16,32,2,4,8,16,32]
    #amount of ram each instance has
    memory=[1.8,3.6,7.2,14.4,28.8,7.5,15,30,60,120,13,26,52,104,208]
    #cost of each instance
    cost=[0.000078931,0.000157709,0.000315263,0.000105708,0.000211263,0.000422374,0.000131709,0.000263265,0.000526374]
    c=0
    timing_info=[]
    total_cost=[]
    timing_high_cpu=[]
    timing_standard=[]
    timing_high_mem=[]
    cost_high_cpu=[]
    cost_standard=[]
    cost_high_mem=[]

    f=open("./config.txt","r")
    for line in f:
        #user will provide information about whether each stage is multi-threaded or not since that flag needs to be changed according to the type of instance and how many vcpus the instance has
        if "Multi-threaded" in line:
            multi_t=line.split('=')
            multi_thread=multi_t[1]
            multi_thread=multi_thread.strip('\n')
            if ',' in multi_thread:
                multi_threading=multi_thread.split(',')
            else:
                multi_threading.append(multi_thread)
    f.close()

    #OAI pipeline
    if input_type == "Image":
        
        #take first image from input bucket and execute it on all instances
        temp1=input_path.rfind('/')
        tmp_path=input_path[:rfind]
        tmp_path_1=input_path[rfind:]
        os.system("gsutil ls "+input_path+" > input.txt")
        f=open("./input.txt","r")
        for line in f:
            if c!=0 and c < 2:
                os.system("gsutil cp -r "+line+" "+tmp_path_1+"/tmp")
            c=c+1
        f.close()
        
        for i in range(len(command)):
            

            command[i].replace("${INPUT_PATH}","/input_"+str(i))
            command[i].replace("${OUTPUT_PATH}","/output_"+str(i))
            
            for j in range(len(memory)):
                if (multi_threading[i])!='NO':
                    multit=command[i].split(' '+multi_threading[i]+' ')
                    multith=multit[1]
                    multithr=multith.split(' ')
                    multithre=multithr[0]
                    command[i]=command[i].replace(multi_threading[i]+" "+str(multithre), multi_threading[i]+" "+str(vcpu[j]))

                f=open("./startup_"+str(vcpu[j])+"_"+str(memory[j])+".sh","w")
                f.write("curl -fsSL https://get.docker.com -o get-docker.sh")
                f.write("sh get-docker.sh")
                f.write("mkdir /input")
                f.write("mkdir /output")
                #copy from output of previous stage
                f.write("gsutil cp -r "+tmp_path+"/tmp/ /input/")
                f.write("sudo docker run -dit "+d_image)
                f.write('sudo docker ps --format "{{.Names}} > /d_name.txt')
                f.write("while IFS= read -r line; do d_name= $line; done < /d_name.txt")
                f.write("sudo docker exec $d_name mkdir /input_"+str(i))
                f.write("sudo docker exec $d_name mkdir /output_"+str(i))
                f.write("sudo docker cp /input/ $d_name:/input_"+str(i))
                #time the docker exec and write it to a file
                f.write("sudo docker exec $d_name "+command[i])
                f.write("sudo docker cp $d_name:/output_"+str(i)+" /output")
                #copy to common output directory so that it can be fed as input to next stage
                f.write("gsutil cp -r /output/ "+output_path)
                #copy timing info to bucket
                f.write("gsutil cp /timing_"+str(i)+"_"+str(j)+".txt "+input_path)
                f.close()

                #launch gcloud vm with the above startup script
                os.system("gcloud compute instances create instance-"+str(i)+"-"+str(j)+" --metadata-from-file startup-script=./startup_"+str(vcpu[j])+"_"+str(memory[j])+".sh")

            #download all timing information from bucket and parse it into timing_info array
            #multipy timing_info array with cost array to get cost array
            for k in range(len(vcpu)):
                total_cost.append(timing_info[k]*cost[k])

        timing_high_cpu=timing_info[:int(len(timing_info))/3]
        cost_high_cpu=total_cost[:int(len(total_cost))/3]
        timing_standard=timing_info[int(len(timing_info))/3:int(len(timing_info))/3+int(len(timing_info))/3]
        cost_standard=total_cost[int(len(total_cost))/3:int(len(total_cost))/3+int(len(total_cost))/3]
        timing_high_mem=timing_info[int(len(timing_info))/3+int(len(timing_info))/3:]
        cost_high_mem=total_cost[int(len(total_cost))/3+int(len(total_cost))/3:]

        fastest_high_cpu=min(timing_high_cpu)
        fastest_standard=min(timing_standard)
        fastest_high_mem=min(timing_high_mem)

        fastest=min(fastest_high_cpu,fastest_standard,fastest_high_mem)
        fastest_index=timing_info.index(fastest)

        cheapest_high_cpu=min(cost_high_cpu)
        cheapest_standard=min(cost_standard)
        cheapest_high_mem=min(cost_high_mem)

        cheapest=min(cheapest_high_cpu,cheapest_standard,cheapest_high_mem)

        cheapest_index=total_cost.index(cheapest)

        print "STAGE: "+str(i+1)
        print "Cheapest Configuration: "+str(vcpu[cheapest_index])+" cores & "+str(memory[cheapest_index])+" GB RAM. Time: "+str(timing_info[cheapest_index])
        print "Fastest Configuration in terms of execution time: "+str(vcpu[fastest_index])+" cores & "+str(memory[fastest_index])+" GB RAM. Time: "+str(timing_info[fastest_index])


    if input_type == "Genomic":

        f=open("./config.txt","r")
        for line in f:
            #gather all information needed to launch dsub job
            #construct dsub command
        f.close()
        os.system('gsutil cp '+input_path+'/'+input_file[i]+' .')
        if file_extension == ".fastq":
            #downsample using seqtk tool
        if file_extension == ".bam" or file_extenssion == ".sam"
            #downsample using DownsampleSam tool

        #move downsampled file to input path
        os.system("gsutil cp downsample_file "+input_path)

        for i in range(len(commands)):

            job_name=[]
            logging_info=[]
            timing_info=[]
            total_cost=[]
            errors=[]
            flag=0
            timing_high_cpu=[]
            timing_standard=[]
            timing_high_mem=[]
            cost_high_cpu=[]
            cost_standard=[]
            cost_high_mem=[]
            speedup_high_cpu=[]
            speedup_standard=[]
            speedup_high_mem=[]
            normalized_speedup_high_cpu=[]
            normalized_speedup_standard=[]
            normalized_speedup_high_mem=[]

	    #execute the dsub job	
	    for j in range(len(memory)):
		find_ram=dsub[i].split('min-ram ')
	        ram_split=find_ram[1]
       		ram_space=ram_split.split(' ')
	        ram=ram_space[0]
	        dsub[i]=dsub[i].replace("min-ram "+str(ram), "min-ram "+str(memory[j]))

	        find_cores=dsub[i].split('min-cores ')
        	cores_split=find_cores[1]
	        cores_space=cores_split.split(' ')
	        cores=cores_space[0]
        	dsub[i]=dsub[i].replace("min-cores "+str(cores), "min-cores "+str(vcpu[j]))

		#if command has multi-threading, replace the multi-threading argument by the number of VCPUs in the instance
	        if (multi_threading[i])!='NO':
                    multit=dsub[i].split(' '+multi_threading[i]+' ')
                    multith=multit[1]
                    multithr=multith.split(' ')
                    multithre=multithr[0]
                    dsub[i]=dsub[i].replace(multi_threading[i]+" "+str(multithre), multi_threading[i]+" "+str(vcpu[j]))

		if i==(len(commands)-1):
			os.system(dsub[i]+"'"+" > dsub_information.txt")
		else:
			os.system(dsub[i]+" > dsub_information.txt")

		#get the name of the job and the log file
	        f=open("./dsub_information.txt","r")
	        for line in f:
                    job_name.append(line.strip('\n'))
                    log=line.split("--")
                    log_info=log[2]
                    log_info=log_info.strip('\n')
                    logging_info.append(log_info)
		f.close()

            time.sleep(240)

            #check whether the job has finished successfully or not
            while True:
                os.system("dstat --provider google --project "+project+" --jobs '"+job_name[flag]+"' --status '*' > dstat_info.txt")
                f=open("./dstat_info.txt","r")
                for line in f:
                    if "Success" in line:
                        flag=flag+1
                        #print i
                    if "fail" in line:
                        flag=flag+1
                        print job_name[flag-1]+" was unsuccessful"
                f.close()

                if flag==(len(vcpu)):
                    break

                time.sleep(30)

            #once the job is done, save the time taken to a file and check the log file for errors
            for j in range(len(vcpu)):
                os.system("gsutil cp "+log_bucket+"/"+job_name[j]+"-stderr.log .")
                os.system("tail -n 3 "+job_name[j]+"-stderr.log | head -n 1 > timing.txt")
                os.system("tail -n 10 "+job_name[j]+"-stderr.log > check_error.txt")	

                #f=open("./"+job_name[j]+"-stderr.log")
                f=open("./check_error.txt")
                for line in f:
                    if (("fail" in line.lower()) and ("memory" in line.lower())) or ("error" in line.lower()):
                            #commonly occuring error present in all log files, but doesn't affect the job
                        if "GPG error" in line:
                            continue
                        print line
                        print "Log file "+job_name[j]+"-stderr.log seems to have an error"
                        print "Skipping this configuration while providing the best configuration"
                        if j in errors:
                            continue
                        else:
                            errors.append(j)
                f.close()

                #calculate the time taken by the machine to execute the job
                f=open("./timing.txt")
                for line in f:
                    timin=line.split('\t')
                    timin[1]=timin[1].strip('\t')
                    timin[1]=timin[1].strip('\n')
                    timing=timin[1]
                    timing=timing[:-1]
                    timing_i=timing.split('m')
                    m_to_s=(float(timing_i[0]))*60
                    s=float(timing_i[1])
                    total_time=m_to_s+s
                    x=total_time*float(cost[j])

                    if "-Xms" in dsub[i]:
                        temp_mem=dsub[i].split("-Xms")
                        java_mem=temp_mem[1][0]
                        if (int(java_mem)) > (memory[j]-2):
                            total_cost.append(0)
                            timing_info.append(0)
                            break
                    #if there are errors in the log file, do not take that instance into account during cost and timing calculation
                    if j in errors:
                        total_cost.append(0)
                        timing_info.append(0)
                    else:
                        total_cost.append(x)
                        timing_info.append(total_time)
                f.close()

            #calculate the timing and cost for different machine types
            timing_high_cpu=timing_info[:int(len(timing_info))/3]
            while True:
                if 0 in timing_high_cpu:
                    timing_high_cpu.remove(0)
                else:
                    break
            #print "Timing High-CPU: ",timing_high_cpu

            cost_high_cpu=total_cost[:int(len(total_cost))/3]
            while True:
                if 0 in cost_high_cpu:
                    cost_high_cpu.remove(0)
                else:
                    break
            #print "Cost High-CPU: ",cost_high_cpu

            timing_standard=timing_info[int(len(timing_info))/3:int(len(timing_info))/3+int(len(timing_info))/3]
            while True:
                if 0 in timing_standard:
                    timing_standard.remove(0)
                else:
                    break
            #print "Timing Standard: ",timing_standard

            cost_standard=total_cost[int(len(total_cost))/3:int(len(total_cost))/3+int(len(total_cost))/3]
            while True:
                if 0 in cost_standard:
                        cost_standard.remove(0)
                else:
                        break
            #print "Cost Standard: ",cost_standard

            timing_high_mem=timing_info[int(len(timing_info))/3+int(len(timing_info))/3:]
            while True:
                if 0 in timing_high_mem:
                    timing_high_mem.remove(0)
                else:
                    break
            #print "Timing High-Mem: ",timing_high_mem

            cost_high_mem=total_cost[int(len(total_cost))/3+int(len(total_cost))/3:]
            while True:
                if 0 in cost_high_mem:
                    cost_high_mem.remove(0)
                else:
                    break
            #print "Cost High-Mem: ",cost_high_mem

            fastest_high_cpu=min(timing_high_cpu)
            fastest_standard=min(timing_standard)
            fastest_high_mem=min(timing_high_mem)

            fastest=min(fastest_high_cpu,fastest_standard,fastest_high_mem)
            fastest_index=timing_info.index(fastest)

            cheapest_high_cpu=min(cost_high_cpu)
            cheapest_standard=min(cost_standard)
            cheapest_high_mem=min(cost_high_mem)

            cheapest=min(cheapest_high_cpu,cheapest_standard,cheapest_high_mem)

            cheapest_index=total_cost.index(cheapest)

            print "STAGE: "+str(i+1)
            print "Cheapest Configuration: "+str(vcpu[cheapest_index])+" cores & "+str(memory[cheapest_index])+" GB RAM. Time: "+str(timing_info[cheapest_index])
            print "Fastest Configuration in terms of execution time: "+str(vcpu[fastest_index])+" cores & "+str(memory[fastest_index])+" GB RAM. Time: "+str(timing_info[fastest_index])
