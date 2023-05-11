#!/bin/bash
#export HOME=/workspace/home/packages2023/
#export PATH=$HOME/.local/bin:$PATH
dir=$(dirname "$0")
root_dir="$(realpath --relative-to=$dir "$dir/../..")"

cd $dir

weight="checkpoint_key_metric=0.8319.pt"
fold=0
task_id=2120

expr_name=out_train

multi_gpu=0

if (($multi_gpu)); then
  python="torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=1234"
  multi_gpu_str="-multi_gpu "
else
  python="python"
  multi_gpu_str=""
fi

cmd="$python \
 $root_dir/dynunet_pipeline/inference.py \
-datalist_path $root_dir/data/config \
-model_folds_dir $root_dir/data/dynunet_trained_models \
-test_files_dir $root_dir/data/test_images/Task2120_regnobetprimix/imagesTs_fold0 \
-fold $fold \
-expr_name ${expr_name} \
-task_id ${task_id} \
-checkpoint ${weight} \
-$multi_gpu_str\
-val_num_workers 0 \
-no-tta_val"

echo $cmd
$cmd
