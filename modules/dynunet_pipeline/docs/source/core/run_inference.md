## Run inference (example for Task2110)
Inference can be run with the following script:

    `monai-inference-from-nnunet-folder/dynunet_pipeline/commands/task2110/infer.sh`

If you changed the trained model checkpoint (.pt file) in the `data/dynunet_trained_models/task2110` folder, 
make sure that in the infer.sh:

    `weight="net_key_metric=0.XXXX.pt`

matches the new model checkpoint name.

The argument `-test_files_dir` is followed by the path to the folder to the `.nii.gz` images to perform the inference on.

This script will run inference on all images in the images folder and save the predictions in a new folder in the
same directory as the script or, if provided, in the directory specified after the `-val_output_dir` argument.

### Registration and Brain Extraction
The provided test images for Task2110 are registered to MNI space and brain-extracted. However, if the image is either not
registered, or not brain-extracted, or neither, the following arguments need to be passed to preprocess the image before passing it 
to the CNN.

#### Registration to MNI template
If the test image is NOT registered to the MNI template and NOT brain extracted, add the following argument:

    `-registration_template_path ../../../data/templates/MNI_152_mri.nii.gz`

On the other hand, if it is NOT registered but IS brain extracted, use the brain extracted template for registration:

    `-registration_template_path ../../../data/templates/MNI_152_mri_hd-bet_stripped.nii.gz`

#### Brain extraction
If the input image is NOT brain extracted, but the model was trained on brain extracted data, 
you can perform brain extraction with HD-BET by adding the arguments:
    
    `-bet`

    `-val_num_workers 0`

The second argument is required because the brain extraction is CUDA based and CUDA with multiprocessing does not work
by default.
This step will happen AFTER the registration to the MNI template, so if both preprocessing steps
(registration and brain extraction) are required, use the `MNI_152_mri.nii.gz` template for registration.
