# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from glob import glob

import torch.distributed as dist
from monai.data import PersistentStagedDataset, DataLoader, load_decathlon_datalist, \
    partition_dataset, PersistentDataset

from transforms import get_task_transforms


def get_datalist(mode,
                 datalist_path,
                 task_id,
                 modalities,
                 fold=0,
                 train_set_path=None,
                 test_files_dir=None,
                 infer_output_dir=None,
                 mni_prior_path=None):

    # for testing, get the datalist by checking which files are available in test_files_dir directory
    if mode == "test":
        assert test_files_dir, f"test_files_dir must be provided, but is {test_files_dir}..."
        datalist = [{'image': p.replace("_0000.nii.gz", ".nii.gz")}
                    for p in sorted(glob(os.path.join(test_files_dir, "*")), key=str.lower)
                    if "_0000.nii.gz" in p]
        assert (len(datalist) > 0), f"No cases found in {test_files_dir} to run inference on..."

        # remove cases from datalist that are already in the inference folder
        files_in_infer_dir = os.listdir(infer_output_dir)
        reduced_datalist = []
        for d in datalist:
            if not os.path.basename(d['image']) in files_in_infer_dir:
                reduced_datalist.append(d)
            else:
                print(f"Found {d['image']} in {infer_output_dir}. Remove from datalist...")

        assert (len(datalist) > 0), f"No cases left to run inference on..."
        datalist = reduced_datalist

    # for prep, train, and validation, get the datalist from the dataset...json file
    else:
        datalist_filepath = os.path.join(datalist_path, "dataset_task{}.json".format(task_id))
        if mode in ["prep", "train"]:
            list_key = f"train_fold{fold}"
        elif mode in ["validation"]:
            list_key = f"validation_fold{fold}"
        else:
            raise Exception(f"mode needs to be 'prep', 'train' or 'validation' but is '{mode}'...")
        datalist = load_decathlon_datalist(datalist_filepath, True, list_key, train_set_path)

    def expand_paths_for_modalities(data_dict, modality_dict, mni_prior_path):
        """
        Expands the "image" entry in data_dict by "image_0000" for first modality, "image_0001" for second modality etc..
        If prior is used, includes the prior as an additional modality...
        """
        for i, mod in modality_dict.items():
            mod_val_str = "{:04d}".format(int(i))
            data_dict["image_" + mod_val_str] = data_dict["image"].replace(".nii.gz", "_" + mod_val_str + ".nii.gz")

        # add the MNI prior to the data dictionary
        if mni_prior_path:
            i = str(int(i) + 1)
            mod_val_str = "{:04d}".format(int(i))
            data_dict["image_" + mod_val_str] = mni_prior_path

        data_dict.pop("image")
        return data_dict

    datalist = [expand_paths_for_modalities(d, modalities, mni_prior_path) for d in datalist]

    return datalist


def get_dataloader(
        datalist,
        transform_params,
        task_id,
        multi_gpu=False,
        mode="train",
        batch_size=1,
        train_num_workers=None,
        val_num_workers=None,
):

    modality_keys = sorted([k for k in datalist[0].keys() if "image_" in k], key=str.lower)

    # for multi-GPU, split datalist into multiple lists
    if multi_gpu:
        datalist = partition_dataset(
            data=datalist,
            shuffle=True if mode in ["prep", "train"] else False,
            num_partitions=dist.get_world_size(),
            even_divisible=True if mode in ["prep", "train"] else False,
        )[dist.get_rank()]

    if mode == "prep":
        prep_load_tfm = get_task_transforms(mode, task_id, modality_keys, **transform_params)
        prep_ds = PersistentStagedDataset(
            new_transform=prep_load_tfm,
            old_transform=None,
            data=datalist,
            cache_dir="./cache_dir",
        )

        data_loader = DataLoader(
            prep_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=train_num_workers,
            collate_fn=lambda x: x,
        )

    elif mode == "train":
        prep_load_tfm = get_task_transforms("prep", task_id, modality_keys, **transform_params)
        new_tfm = get_task_transforms(mode, task_id, modality_keys, **transform_params)

        train_ds = PersistentStagedDataset(
            new_transform=new_tfm,
            old_transform=prep_load_tfm,
            data=datalist,
            cache_dir="./cache_dir",
        )
        data_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=train_num_workers,
        )

    elif mode in ["validation", "test"]:
        tfm = get_task_transforms(mode, task_id, modality_keys, **transform_params)

        # no caching for testing set
        cache_dir = "./cache_dir" if mode == "validation" else None

        val_ds = PersistentDataset(
            transform=tfm,
            data=datalist,
            cache_dir=cache_dir,
        )

        data_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=val_num_workers, #args.val_num_workers,  # because of the brain-extraction transform, multiprocessing cannot be used, since subprocesses cannot initialize their own CUDA processes. #RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method TODO: try the 'spawn' method: mp.set_start_method("spawn")
        )

    else:
        raise ValueError(f"mode should be train, validation or test.")

    return data_loader
