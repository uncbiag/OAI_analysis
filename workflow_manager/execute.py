f=open("./config.txt","r")
for line in f:
    if "Environment" in line:
        temp1=line.split('Environment=')
        environ1=temp1[1].strip()
        break

execute_path=os.path.abspath("execute.py")
execute_path=execute_path.replace("execute.py","")

if environ1=='Local':
    os.system("python "+execute_path+"/local.py")

if environ1=='GCP':
    os.system("python "+execute_path+"/gcp_downsampling.py")
    os.system("python "+execute_path+"/gcp.py")

if environ1=='AWS':
    os.system("python "+execute_path+"/aws_downsampling.py")
    os.system("python "+execute_path+"/aws.py")

