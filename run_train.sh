#!/bin/bash -e

# Make sure GPUs are up
if [ $SLURM_LOCALID -eq 0 ] ; then
    rocm-smi
fi
sleep 2
echo 1111
# MIOPEN needs some initialisation for the cache as the default location
# does not work on LUMI as Lustre does not provide the necessary features.
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH


if [ $SLURM_LOCALID -eq 0 ] ; then
    rm -rf $MIOPEN_USER_DB_PATH
    mkdir -p $MIOPEN_USER_DB_PATH
fi
sleep 2
export NCCL_IB_DISABLE=1
# export TRITON_DEBUG=1

# Set interfaces to be used by RCCL.
# This is needed as otherwise RCCL tries to use a network interface it has
# no access to on LUMI.
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3
source /opt/miniconda3/bin/activate
conda activate pytorch
export PYTHONPATH=/scratch/$project_id/$USER/nanotron/preview_torch:$PYTHONPATH
export HF_DATASETS_CACHE=/scratch/$project_id/$USER/nanotron/hf_datasets_cache

# Report affinity to check gpu visibility
echo "Rank $SLURM_PROCID --> $(taskset -p $$); GPU $ROCR_VISIBLE_DEVICES"
echo "Rank $SLURM_PROCID on node $SLURMD_NODENAME has GPUs $ROCR_VISIBLE_DEVICES"

# Maybe not all of them are needed ........
export MASTER_ADDR=$(python get_master.py "$SLURM_NODELIST")
echo "$MASTER_ADDR"
export MASTER_PORT="$((${SLURM_JOB_ID} % 10000 + 10000))"
export WORLD_SIZE=$((8*$SLURM_NNODES))
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID 
echo "Node count: $SLURM_NNODES"
echo "World size: $WORLD_SIZE"
echo "Node rank: $SLURM_PROCID"
echo "Local rank: $SLURM_LOCALID"
export OMP_NUM_THREADS=7
export CUDA_DEVICE_MAX_CONNECTIONS=1
# Run application
# python -u testflash.py
python -u -m torch.distributed.run \
    --nproc_per_node=8 \
    --nnodes=${SLURM_NNODES} \
    --rdzv_id=$SLURM_JOBID \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --node_rank $SLURM_PROCID \
    run_train.py --config-file examples/config_tiny_llama.yaml