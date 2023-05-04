#!/bin/bash
#export HOME=/workspace/home/packages2023/
#export PATH=$HOME/.local/bin:$PATH
dir=$(dirname "$0")

cd $dir

out_folder_name=out_train
lr=1e-2
fold=0
task=2120
prior_path="../../../data/train_images/Task2120_regnobetprimix/argmax_fuzzy_prior_MIX_fold0_tr+val_GIF.nii.gz"

data_dir=../../../data/train_images/

#python \
#python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=1234 \
cmd="python \
../../train.py \
-datalist_path ../../../data/dynunet_trained_models/config \
-fold $fold \
-model_folds_dir
.
-train_num_workers 6 \
-interval 1 \
-learning_rate $lr \
-max_epochs 4 \
-task_id ${task} \
-pos_sample_num 2 \
-neg_sample_num 1 \
-expr_name ${out_folder_name} \
-root_dir ${data_dir} \
-amp \
-no-tta_val
-prior_path ${prior_path} \
-resume_latest_checkpoint \
"

echo $cmd
$cmd
