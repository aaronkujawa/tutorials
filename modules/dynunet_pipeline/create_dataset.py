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
from monai.data import PersistentStagedDataset, DataLoader, load_decathlon_datalist, load_decathlon_properties, \
    partition_dataset, PersistentDataset

from task_params import task_name
from transforms import get_task_transforms


def get_data(args, batch_size=1, mode="train"):
    # get necessary parameters:
    fold = args.fold
    task_id = args.task_id
    root_dir = args.root_dir
    datalist_path = args.datalist_path
    test_files_dir = args.test_files_dir if hasattr(args, 'test_files_dir') else None
    dataset_path = os.path.join(root_dir, task_name[task_id])

    preproc_out_dir = args.preproc_out_dir if hasattr(args, "preproc_out_dir") else None

    use_mni_prior = True if (hasattr(args, "mni_prior_path") and args.mni_prior_path) else False

    transform_params = (args.pos_sample_num, args.neg_sample_num)
    multi_gpu_flag = args.multi_gpu

    if mode == "test":
        list_key = "test"
    elif mode == "prep":
        list_key = "{}_fold{}".format("train", fold)
    else:
        list_key = "{}_fold{}".format(mode, fold)
    datalist_name = "dataset_task{}.json".format(task_id)

    property_keys = [
        "name",
        "description",
        "reference",
        "licence",
        "tensorImageSize",
        "modality",
        "labels",
        "numTraining",
        "numTest",
    ]

    properties = load_decathlon_properties(os.path.join(datalist_path, datalist_name), property_keys)

    if mode == "test":
        datalist = [{'image': p.replace("_0000.nii.gz", ".nii.gz")}
                    for p in sorted(glob(os.path.join(test_files_dir, "*")), key=str.lower)
                    if "_0000.nii.gz" in p]
        assert (len(datalist) > 0), f"No cases found in {test_files_dir} to run inference on..."

        # remove cases from datalist that are already in the inference folder
        files_in_infer_dir = os.listdir(args.infer_output_dir)
        reduced_datalist = []
        for d in datalist:
            if not os.path.basename(d['image']) in files_in_infer_dir:
                reduced_datalist.append(d)
            else:
                print(f"Found {d['image']} in {args.infer_output_dir}. Remove from datalist...")

        assert (len(datalist) > 0), f"No cases left to run inference on..."
        datalist = reduced_datalist


    else:
        datalist = load_decathlon_datalist(
            os.path.join(datalist_path, datalist_name), True, list_key, dataset_path
        )

    # the datalist needs to be extended by _0000 for first modality, _0001 for second modality etc...
    modalities = properties['modality']

    def expand_paths_for_modalities(data_dict, modality_dict, mni_prior_path):

        for i, mod in modality_dict.items():
            mod_val_str = "{:04d}".format(int(i))
            data_dict["image_"+mod_val_str] = data_dict["image"].replace(".nii.gz", "_"+mod_val_str+".nii.gz")

        # add the MNI prior to the data dictionary
        if mni_prior_path:
            i = str(int(i)+1)
            mod_val_str = "{:04d}".format(int(i))
            data_dict["image_" + mod_val_str] = mni_prior_path

        data_dict.pop("image")
        return data_dict

    datalist = [expand_paths_for_modalities(d, modalities, use_mni_prior) for d in datalist]
    modality_keys = sorted([k for k in datalist[0].keys() if "image_" in k], key=str.lower)
    if mode == "prep":
        if multi_gpu_flag:
            datalist = partition_dataset(
                data=datalist,
                shuffle=True,
                num_partitions=dist.get_world_size(),
                even_divisible=True,
            )[dist.get_rank()]

        prep_load_tfm = get_task_transforms(mode, task_id, modality_keys, *transform_params)

        prep_ds = PersistentStagedDataset(
            new_transform=prep_load_tfm,
            old_transform=None,
            data=datalist,  # shorten the list for debugging
            cache_dir="./cache_dir",
        )

        data_loader = DataLoader(
            prep_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.train_num_workers,
            collate_fn=lambda x: x,
        )

    elif mode == "train":
        if multi_gpu_flag:
            datalist = partition_dataset(
                data=datalist,
                shuffle=True,
                num_partitions=dist.get_world_size(),
                even_divisible=True,
            )[dist.get_rank()]

        prep_load_tfm = get_task_transforms("prep", task_id, modality_keys, *transform_params)
        new_tfm = get_task_transforms(mode, task_id, modality_keys, *transform_params)

        train_ds = PersistentStagedDataset(
            new_transform=new_tfm,
            old_transform=prep_load_tfm,
            data=datalist,  # shorten the list for debugging
            cache_dir="./cache_dir",
        )
        data_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.train_num_workers,
        )

    elif mode in ["validation", "test"]:
        if multi_gpu_flag:
            datalist = partition_dataset(
                data=datalist,
                shuffle=False,
                num_partitions=dist.get_world_size(),
                even_divisible=False,
            )[dist.get_rank()]

        tfm = get_task_transforms(mode, task_id, modality_keys, *transform_params)

        if mode == "validation":
            cache_dir = "./cache_dir"
        elif mode == "test":  # No caching for testing
            cache_dir = None

        val_ds = PersistentDataset(
            transform=tfm,
            data=datalist,  # shorten the list for debugging
            cache_dir=cache_dir,
        )

        data_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.val_num_workers, #args.val_num_workers,  # because of the brain-extraction transform, multiprocessing cannot be used, since subprocesses cannot initialize their own CUDA processes. #RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method TODO: try the 'spawn' method: mp.set_start_method("spawn")
        )
    else:
        raise ValueError(f"mode should be train, validation or test.")

    return properties, data_loader
