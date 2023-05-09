import os
# assemble the docker command
# get UID do pass this to docker command, since docker container will otherwise create files as
# root (which makes them difficult to delete afterwards)
try: # the following command doesn't work on windows
  UID = os.getuid()
except: # hence, on windows, assume a default user id
  UID = 1000

script_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(script_path, "..", "..")

model_folds_dir = os.path.join(root_path, "data", "dynunet_trained_models")
test_files_dir = os.path.join(root_path, "data", "test_images", "Task2120_regnobetprimix", "imagesTs_fold0")
config_dir = os.path.join(root_path, "data", "config")
out_dir = os.path.join(script_path, "inference_output")

if not os.path.isdir(out_dir):
  os.makedirs(out_dir, exist_ok=True)

cmd = f"""docker run 
-u {UID} 
-v {model_folds_dir}:/workspace/model_folds_dir 
-v {test_files_dir}:/workspace/imagesTs 
-v {config_dir}:/workspace/config 
-v {out_dir}:/home/fast_parc/out
--ipc=host 
--gpus 0 
aaronkujawa/fast_parcellation:latest
--datalist_path /workspace/config
--model_folds_dir /workspace/model_folds_dir 
--test_files_dir /workspace/imagesTs 
--val_output_dir /home/fast_parc/out
--fold 0
--expr_name out_train
--task_id 2120
--checkpoint checkpoint_key_metric=0.8319.pt
--val_num_workers 0 
--no-tta_val
""".replace("\n", " ")

#--registration_template_path /workspace/model_folds_dir/task2120/runs_2120_fold0_out_train/argmax_fuzzy_prior_MIX_fold0_tr+val_GIF.nii.gz
#--do_brain_extraction


print(cmd)

os.system(cmd)