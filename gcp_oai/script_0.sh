#!/bin/bash
python3.6 /home/OAI_analysis/execute.py --month 12 --OAI_data gs://oai_bucket --cloud google --size 3 --rank 0 --OAI_data_sheet /home/OAI_analysis/tmp.csv --OAI_results ${OUTPUT_PATH}
