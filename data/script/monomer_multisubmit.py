import os
import time
import pandas as pd

import math

from subprocess import call

if __name__ == '__main__':
    ### 'https://stackoverflow.com/questions/19431990/submitting-jobs-using-python' ###
    # job should be submitted from the global node
    parallel_num = 48
    total_number_of_data = pd.read_csv('/home/sk77/PycharmProjects/publish/OMG/data/version.smi', sep=' ').shape[0]
    print(total_number_of_data, flush=True)
    parallel_number_of_data = math.ceil(total_number_of_data / parallel_num)
    file = '/home/sk77/PycharmProjects/publish/OMG/data/script/preprocess.py'

    # parallel processing
    for num in range(36, 48):
        start_idx = num * parallel_number_of_data
        if num == parallel_num - 1:
            end_idx = total_number_of_data
        else:
            end_idx = (num + 1) * parallel_number_of_data

        # write bash scripts
        job_call = f'/home/sk77/.conda/envs/gpu_test/bin/python3 %s ' \
                   f'/home/sk77/PycharmProjects/publish/OMG/data/OMG_monomer_process_batch {num} {start_idx} {end_idx}\n'
        qsub_call = 'qsub run_%d.sh'
        bash_name = 'run_%d.sh'

        bash_lines = [
            '#!/bin/csh\n',
            '#$ -N OMG_%d\n' % num,
            '#$ -cwd\n',
            '#$ -j y\n',
            '#$ -o $JOB_ID.out\n',
            '#$ -pe orte 1\n',
            '#$ -l hostname=compute-0-0.local\n',
            '#$ -q all.q\n',
        ]

        with open(bash_name % num, 'w') as bash_file:
            bash_file.writelines(bash_lines + [job_call % file])

        # submit a job
        call(qsub_call % num, shell=True)

    # delete .sh file
    sh_files = [j for j in os.listdir(".") if '.sh' in j]
    for file in sh_files:
        os.remove(file)
