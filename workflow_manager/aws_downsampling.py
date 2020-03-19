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
    os.system("aws s3 ls "+input_path+" > input.txt")
    f=open("./input.txt","r")
    for line in f:
        if c!=0 and c < 2:
            os.system("aws s3 cp -r "+line+" "+tmp_path_1+"/tmp")
        c=c+1
    f.close()

if input_type == "Genomic":

    f=open("./config.txt","r")
    for line in f:
        #gather all information needed to launch vm
    f.close()
    os.system('aws s3 cp '+input_path+'/'+input_file[i]+' .')
    if file_extension == ".fastq":
        #downsample using seqtk tool
    if file_extension == ".bam" or file_extenssion == ".sam"
        #downsample using DownsampleSam tool

    #move downsampled file to input path
    os.system("aws s3 cp downsample_file "+input_path)

    
for i in range(len(command)):
    

    command[i].replace("${INPUT_PATH}","/input_"+str(i))
    command[i].replace("${OUTPUT_PATH}","/output_"+str(i))
    
    for j in range(len(machines)):
        if (multi_threading[i])!='NO':
            multit=command[i].split(' '+multi_threading[i]+' ')
            multith=multit[1]
            multithr=multith.split(' ')
            multithre=multithr[0]
            command[i]=command[i].replace(multi_threading[i]+" "+str(multithre), multi_threading[i]+" "+str(vcpu[j]))

        f=open("./startup_"+str(machines[j])+".sh","w")
        f.write("curl -fsSL https://get.docker.com -o get-docker.sh")
        f.write("sh get-docker.sh")
        f.write("mkdir /input")
        f.write("mkdir /output")
        #copy from output of previous stage
        f.write("aws s3 cp -r "+tmp_path+"/tmp/ /input/")
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
        f.write("aws s3 cp -r /output/ "+output_path)
        #copy timing info to bucket
        f.write("aws s3 cp /timing_"+str(i)+"_"+str(j)+".txt "+input_path)
        f.close()

        #launch aws vm with the above startup script
        os.system("aws ec2 run-instances –image-id "+image_id+" --count 1--instance-type "+machines[j]+" --key-name "+key+" --subnet-id "+subnet_id+" –security-group-ids "+security_group+" --user-data ./startup_"+str(machines[j])+".sh")

    #download all timing information from bucket and parse it into timing_info array
    #multipy timing_info array with cost array to get cost array
    for k in range(len(machines)):
        total_cost.append(timing_info[k]*cost[k])


