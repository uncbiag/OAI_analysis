docker run -it uray10/oai:secondversion
cd /home/OAI_analysis
ls -l > out.txt
gsutil cp out.txt gs://oai_bucket
