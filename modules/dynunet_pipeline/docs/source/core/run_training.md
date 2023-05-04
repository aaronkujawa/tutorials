## Run training (example for Task2110)
Training can be run by running this script:

    `monai-inference-from-nnunet-folder/dynunet_pipeline/commands/task2110/train.sh`

This command will run training and save model checkpoints under:

    `monai-inference-from-nnunet-folder/dynunet_pipeline/commands/task2110/runs_2110_fold0_out_train`

If you want to use the newly trained model, copy this folder to: 

    `monai-inference-from-nnunet-folder/data/dynunet_trained_models/task2110`

But for now we will continue with the pretrained model that is saved under that location already.