#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --time=1:00:00
#SBATCH --job-name="nlu_proj"

export MULTIFC_DIR=/Users/jesse/nyu_ms_ds/1012/nlu-finalproject-2020/Jesse/multi-fc


# Module setup (Run every time you get on a machine)
module purge
module load anaconda3/5.3.1
module load cuda/10.0.130
module load gcc/6.3.0

# Replace with your NetID
NETID=js11133
source activate /scratch/${NETID}/nlu/env

# === Set up transformers === #
export PYTORCH_TRANSFORMERS_CACHE=/scratch/${NETID}/nlu/cache
conda install pytorch torchvision -c pytorch

cd /scratch/${NETID}/nlu/nlu-finalproject-2020/transformers
pip install -r ./examples/requirements.txt
pip install boto3 filelock requests tqdm sentencepiece sacremoses tokenizers pandas

export TASK_NAME=MULTIFC
export PYTHONPATH=/scratch/${NETID}/nlu/nlu-finalproject-2020/transformers/src:$PYTHONPATH
python /scratch/${NETID}/nlu/nlu-finalproject-2020/transformers/examples/run_glue.py \
    --data_dir /scratch/js11133/nlu/multi-fc \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name ${TASK_NAME} \
    --max_seq_length 512 \
    --do_train \
    --do_eval \
    --overwrite_output_dir \
    --learning_rate 2e-5 \
    --output_dir /scratch/${NETID}/nlu/multi-fc-out/ \
    --seed 5