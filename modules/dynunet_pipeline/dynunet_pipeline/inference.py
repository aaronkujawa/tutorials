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

import json
import logging
import os
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch
import torch.distributed as dist
from monai.data import load_decathlon_properties
from monai.inferers import SlidingWindowInferer
from torch.nn.parallel import DistributedDataParallel

from create_dataset import get_dataloader, get_datalist
from create_network import get_network, get_kernels_strides
from inferrer import DynUNetInferrer


def setup_root_logger():
    logger = logging.root
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt=None)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_prior_path(model_folder_path, train_args_dict):
    prior_path = os.path.join(model_folder_path, os.path.basename(train_args_dict["prior_path"])) if (
                "prior_path" in train_args_dict and train_args_dict["prior_path"]) else None
    if prior_path and os.path.isfile(prior_path):
        print(f"Found prior: {prior_path}")
    elif prior_path and not os.path.isfile(prior_path):
        raise Exception(f"Prior file not found: {prior_path}")
    else:
        print("No prior provided with --prior_path ...")

    return prior_path


def read_task_params(task_params_filepath, task_id):
    with open(task_params_filepath, "r") as f:
        task_params_all_tasks = json.load(f)
    # pick values of the relevant task_id only
    task_params = {k:v[task_id] for k, v in task_params_all_tasks.items()}
    return task_params


def inference(args):
    # load hyper parameters
    task_id = args.task_id
    checkpoint = args.checkpoint
    model_folder_path = os.path.join(args.model_folds_dir, "task" + task_id, "runs_{}_fold{}_{}".format(
        args.task_id, args.fold, args.expr_name))
    val_output_dir = os.path.join(args.val_output_dir, "task" + task_id, "runs_{}_fold{}_{}".format(
        args.task_id, args.fold, args.expr_name))
    sw_batch_size = args.sw_batch_size
    window_mode = args.window_mode
    eval_overlap = args.eval_overlap
    amp = args.amp
    tta_val = args.tta_val
    multi_gpu_flag = args.multi_gpu
    local_rank = args.local_rank
    datalist_path = args.datalist_path

    if multi_gpu_flag:
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda")

    # read the file that stored the arguments of the model training (we need info on a parameter of the normalization
    # transform and whether a prior was used)
    with open(os.path.join(model_folder_path, "training_params.json"), "r") as f:
        train_args_dict = json.load(f)

    # load dataset properties
    datalist_filepath = os.path.join(datalist_path, f"dataset_task{task_id}.json")
    task_params_filepath = os.path.join(datalist_path, "task_params.json")
    task_params = read_task_params(task_params_filepath, task_id)
    properties = load_decathlon_properties(datalist_filepath, property_keys=["modality", "labels", "tensorImageSize"])

    # prior should have been moved to model directory --> adjust path
    prior_path = get_prior_path(model_folder_path, train_args_dict)

    # define output directories
    infer_output_dir = os.path.join(val_output_dir, task_params["task_name"])
    preproc_out_dir = os.path.join(infer_output_dir, "preprocessed")
    os.makedirs(infer_output_dir, exist_ok=True)

    # get datalist
    datalist_testing = get_datalist("test", datalist_path, task_id, properties['modality'],
                                    test_files_dir=args.test_files_dir,
                                    infer_output_dir=infer_output_dir,
                                    prior_path=prior_path)

    # calculate kernel sizes and strides
    kernels, strides = get_kernels_strides(task_params["patch_size"], task_params["spacing"])

    # parameters used by transforms
    transform_params = {
                        "patch_size": task_params["patch_size"],
                        "use_nonzero": train_args_dict["use_nonzero"],
                        "registration_template_path": args.registration_template_path if hasattr(args, "registration_template_path") else None,
                        "preproc_out_dir": preproc_out_dir,
                        "bet": args.bet if hasattr(args, "registration_template_path") else None,
                        "use_prior": True if prior_path else False
                        }

    # parameters used by dataloaders
    dataloader_params = {
        "val_num_workers": args.val_num_workers,
    }

    # get dataloader
    test_loader = get_dataloader(datalist_testing,
                                 transform_params,
                                 task_id,
                                 multi_gpu_flag,
                                 mode="test",
                                 batch_size=1,
                                 **dataloader_params,
                                 )

    net = get_network(
        n_classes=len(properties["labels"]),
        n_in_channels=len(properties["modality"]),
        kernels=kernels,
        strides=strides,
        deep_supr_num=task_params["deep_supr_num"],
        prior_path=prior_path,
        pretrain_path=model_folder_path,
        checkpoint=checkpoint)
    net = net.to(device)

    if multi_gpu_flag:
        net = DistributedDataParallel(module=net, device_ids=[device], find_unused_parameters=True)

    net.eval()

    inferrer = DynUNetInferrer(
        device=device,
        val_data_loader=test_loader,
        network=net,
        output_dir=infer_output_dir,
        num_classes=len(properties["labels"]),
        inferer=SlidingWindowInferer(
            roi_size=task_params["patch_size"],
            sw_batch_size=sw_batch_size,
            overlap=eval_overlap,
            mode=window_mode,
            device="cpu",
            progress=True,
        ),
        amp=amp,
        tta_val=tta_val,
    )

    inferrer.run()


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-fold", "--fold", type=int, default=0, help="0-5")
    parser.add_argument("-task_id", "--task_id", type=str, default="02", help="task 01 to 10")
    parser.add_argument(
        "-root_dir",
        "--root_dir",
        type=str,
        default="/workspace/data/medical/",
        help="dataset path",
    )
    parser.add_argument(
        "-model_folds_dir",
        "--model_folds_dir",
        type=str,
        default="",
        help="Path to folder that contains subfolders task01, task02... with trained models",
    )
    parser.add_argument(
        "-val_output_dir",
        "--val_output_dir",
        type=str,
        default="",
        help="Path to folder that contains subfolders task01, task02... where to store inference results",
    )
    parser.add_argument(
        "-expr_name",
        "--expr_name",
        type=str,
        default="expr",
        help="the suffix of the experiment's folder",
    )
    parser.add_argument(
        "-datalist_path",
        "--datalist_path",
        type=str,
        default="config/",
        help="where to find the task*/dataset.json"
    )
    parser.add_argument(
        "-test_files_dir",
        "--test_files_dir",
        type=str,
        default="",
        help="where to look for the *.nii.gz files to use for inference."
    )
    parser.add_argument(
        "-registration_template_path",
        "--registration_template_path",
        type=str,
        default="",
        help="Location of template for affine registration during pre-processing."
    )
    parser.add_argument(
        "-val_num_workers",
        "--val_num_workers",
        type=int,
        default=0,
        help="the num_workers parameter of validation dataloader.",
    )
    parser.add_argument(
        "-eval_overlap",
        "--eval_overlap",
        type=float,
        default=0.5,
        help="the overlap parameter of SlidingWindowInferer.",
    )
    parser.add_argument(
        "-sw_batch_size",
        "--sw_batch_size",
        type=int,
        default=1,
        help="the sw_batch_size parameter of SlidingWindowInferer.",
    )
    parser.add_argument(
        "-window_mode",
        "--window_mode",
        type=str,
        default="gaussian",
        choices=["constant", "gaussian"],
        help="the mode parameter for SlidingWindowInferer.",
    )
    parser.add_argument(
        "-cache_rate",
        "--cache_rate",
        type=float,
        default=1.0,
        help="the cache_rate parameter of CacheDataset.",
    )
    parser.add_argument(
        "-checkpoint",
        "--checkpoint",
        type=str,
        default=None,
        help="the filename of weights.",
    )

    parser.add_argument('-amp', '--amp', dest='amp', action='store_true', help="whether to use automatic mixed precision.")
    parser.add_argument('-no-amp', '--no-amp', dest='amp', action='store_false')
    parser.set_defaults(amp=True)

    parser.add_argument('-tta_val', '--tta_val', dest='tta_val', action='store_true', help="whether to use test time augmentation.")
    parser.add_argument('-no-tta_val', '--no-tta_val', dest='tta_val', action='store_false')
    parser.set_defaults(tta_val=True)

    parser.add_argument('-multi_gpu', '--multi_gpu', dest='multi_gpu', action='store_true', help="whether to use multiple GPUs for training.")
    parser.add_argument('-no-multi_gpu', '--no-multi_gpu', dest='multi_gpu', action='store_false')
    parser.set_defaults(multi_gpu=False)

    parser.add_argument('-bet', '--bet', dest='bet', action='store_true', help="whether to perform perform brain extraction during preprocessing.")
    parser.add_argument('-no-bet', '--no-bet', dest='bet', action='store_false')
    parser.set_defaults(bet=False)

    parser.add_argument("-local_rank", "--local_rank", type=int, default=0)
    args = parser.parse_args()
    setup_root_logger()
    inference(args)
