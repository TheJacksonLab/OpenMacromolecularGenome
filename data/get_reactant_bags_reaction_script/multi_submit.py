import os
import time

from subprocess import call

if __name__ == '__main__':
    ### 'https://stackoverflow.com/questions/19431990/submitting-jobs-using-python' ###
    # job should be submitted from the global node
    # get files to submit - files should be in the same directory as the 'multi_submit.py'
    files = [j for j in os.listdir(".") if 'get_reactant_bag_' in j and '.py' in j]
    # files = ['get_reactant_bag_reaction_3.py', 'get_reactant_bag_reaction_4.py', 'get_reactant_bag_reaction_6.py',
    #          'get_reactant_bag_reaction_7.py', 'get_reactant_bag_other_reactions.py']
    # files = ['get_reactant_bag_other_reactions.py']

    # write bash scripts
    job_call = '/home/sk77/.conda/envs/gpu_test/bin/python3 %s ' \
               '../OMG_polymers_batch ../OMG_monomers_v3.csv\n'
    qsub_call = 'qsub run_%d.sh'
    bash_name = 'run_%d.sh'
    for idx, file in enumerate(files):
        bash_lines = [
            '#!/bin/csh\n',
            '#$ -N OMG_%d\n' % idx,
            '#$ -cwd\n',
            '#$ -j y\n',
            '#$ -o $JOB_ID.out\n',
            '#$ -pe orte 1\n',
            '#$ -l hostname=compute-0-0.local\n',
            '#$ -q all.q\n',
            'export PYTHONPATH=/home/sk77/PycharmProjects/polymer\n'
        ]
        with open(bash_name % idx, 'w') as bash_file:
            bash_file.writelines(bash_lines + [job_call % file])

        # submit a job
        call(qsub_call % idx, shell=True)

    # delete .sh file
    sh_files = [j for j in os.listdir(".") if '.sh' in j]
    for file in sh_files:
        os.remove(file)
