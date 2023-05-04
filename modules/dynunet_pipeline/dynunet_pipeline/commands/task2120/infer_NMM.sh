# please replace the weight variable into your actual weight
#export HOME=/workspace/home/packages2023/
#export PATH=$HOME/.local/bin:$PATH

dir=$(dirname "$0")

cd $dir

weight="checkpoint_key_metric=0.8319.pt"
fold=0
task_id=2120

expr_name=out_train

#python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=1234 \
cmd="python \
 ../../../inference.py \
-datalist_path ../../../data/dynunet_trained_models/config \
-model_folds_dir ../../../data/dynunet_trained_models \
-test_files_dir /mnt/nas1/Datasets/NMM_BrainParc/NMM_BrainParc_clean/images \
-fold $fold \
-expr_name ${expr_name} \
-task_id ${task_id} \
-checkpoint ${weight} \
-val_num_workers 0 \
-no-tta_val
-registration_template_path /home/aaron/Dropbox/KCL/Projects/monai_inference_from_nnunet_folder/data/templates/MNI_152_mri.nii.gz \
"

echo $cmd
$cmd
