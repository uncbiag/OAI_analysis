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

