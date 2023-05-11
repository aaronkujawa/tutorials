#!/bin/bash
#export HOME=/workspace/home/packages2023/
#export PATH=$HOME/.local/bin:$PATH
dir=$(dirname "$0")
root_dir="$(realpath --relative-to=$dir "$dir/../..")"

cd $dir

out_folder_name=out_train
lr=1e-2
fold=0
task=2120
max_epochs=1000
prior_path="$root_dir/data/train_images/Task2120_regnobetprimix/argmax_fuzzy_prior_MIX_fold0_tr+val_GIF.nii.gz"

data_dir=$root_dir/data/train_images/

multi_gpu=0

if (($multi_gpu)); then
  python="python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=1234"
  multi_gpu_str="-multi_gpu "
else
  python="python"
  multi_gpu_str=""
fi

cmd="$python \
$root_dir/dynunet_pipeline/train.py \
-datalist_path $root_dir/data/config \
-fold $fold \
-model_folds_dir . \
-train_num_workers 6 \
-interval 5 \
-learning_rate $lr \
-max_epochs ${max_epochs} \
-task_id ${task} \
-pos_sample_num 2 \
-neg_sample_num 1 \
-expr_name ${out_folder_name} \
-root_dir ${data_dir} \
$multi_gpu_str\
-amp \
-no-tta_val \
-prior_path ${prior_path} \
-resume_latest_checkpoint \
"

 echo $cmd
 $cmd |& tee log_train.txt