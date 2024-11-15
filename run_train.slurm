#!/bin/bash -e
#SBATCH --job-name=run_train
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-gpu=7
#SBATCH --output="output_%x_%j.out"
#SBATCH --error="output_%x_%j.err"
#SBATCH --partition=standard-g
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --account=${project_id}

# module load LUMI  # Which version doesn't matter, it is only to get the container.
export SIFPYTORCH=/appl/local/containers/easybuild-sif-images/lumi-pytorch-rocm-6.2.0-python-3.10-pytorch-v2.3.0-dockerhash-187f41102477.sif
# Optional: Inject the environment variables for NCCL debugging into the container.   
# This will produce a lot of debug output!     
# export SINGULARITYENV_NCCL_DEBUG=INFO
# export SINGULARITYENV_NCCL_DEBUG_SUBSYS=INIT,COLL
c=fe
MYMASKS="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"
Nodes=$SLURM_NNODES
echo "Launching training..."
Nodes=$SLURM_NNODES

srun --jobid $SLURM_JOBID \
    --cpu-bind=mask_cpu:$MYMASKS \
singularity exec \
         -B /usr/bin/scontrol \
         -B $(pwd) \
         -B /var/spool/slurmd \
         -B /opt/cray \
         -B /usr/lib64/libcxi.so.1 \
         -B /usr/lib64/slurm/libslurmfull.so \
         -B /etc/slurm \
         -B /usr/lib64/libjansson.so.4 \
         $SIFPYTORCH ./run_train.sh

if [ $? -ne 0 ]; then
    echo "Training failed"
    exit 1
fi
echo "Training complete!"