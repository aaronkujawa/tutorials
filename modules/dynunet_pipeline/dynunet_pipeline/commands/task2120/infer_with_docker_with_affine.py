import os
# assemble the docker command
# get UID do pass this to docker command, since docker container will otherwise create files as
# root (which makes them difficult to delete afterwards)
try: # the following command doesn't work on windows
  UID = os.getuid()
except: # hence, on windows, assume a default user id
  UID = 1000

model_folds_dir = os.path.realpath("../../../data/dynunet_trained_models")
test_files_dir = os.path.realpath("../../../data/test_images/Task2120_regnobetprimix/imagesTs_fold0")
out_dir = os.path.realpath("./inference_output")
templates_dir = os.path.realpath("../../../data/templates")

if not os.path.isdir(out_dir):
  os.makedirs(out_dir, exist_ok=True)

cmd = f"""docker run 
-u {UID} 
-v {model_folds_dir}:/workspace/model_folds_dir 
-v {test_files_dir}:/workspace/imagesTs 
-v {out_dir}:/home/fast_parc/out
-v {templates_dir}:/workspace/templates
--ipc=host 
--gpus 0 
aaronkujawa/fast_parcellation:latest
--datalist_path /workspace/model_folds_dir/config
--model_folds_dir /workspace/model_folds_dir 
--test_files_dir /workspace/imagesTs 
--val_output_dir /home/fast_parc/out
--fold 0
--expr_name out_train
--task_id 2120
--checkpoint checkpoint_key_metric=0.8319.pt
--val_num_workers 0 
--no-tta_val
--registration_template_path /workspace/templates/MNI_152_mri.nii.gz
""".replace("\n", " ")

#--do_brain_extraction


print(cmd)

os.system(cmd)