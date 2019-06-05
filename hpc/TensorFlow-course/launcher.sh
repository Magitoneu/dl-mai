#!/bin/bash
# @ job_name= multiGPUproof
# @ initialdir= .
# @ output= 1GPU-%j.out
# @ error= %j.err
# @ total_tasks= 1
# @ gpus_per_node= 2
# @ cpus_per_task= 1
# @ features= k80
# @ wall_clock_limit= 00:01:30

##SBATCH --reservation=PATC_DL2

#::Code for running in MT with Python 2.7.12
module purge
module load merovingian
module load K80 cuda/8.0 mkl/2017.1 CUDNN/5.1.10-cuda_8.0 intel-opencl/2016 python/3.6.0+_ML
python mnist_mgpu.py
#python /gpfs/home//nct01/nct01022/multiGPUproof.py

