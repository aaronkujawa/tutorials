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
import shutil
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from glob import glob

import ignite.distributed as idist
import torch
import torch.distributed as dist
from ignite.engine import Events
from monai.config import print_config
from monai.data import load_decathlon_properties
from monai.handlers import CheckpointSaver, LrScheduleHandler, MeanDice, StatsHandler, ValidationHandler, from_engine
from monai.handlers.checkpoint_saver import Checkpoint
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.utils import set_determinism
from torch.nn.parallel import DistributedDataParallel

from create_dataset import get_dataloader, get_datalist
from create_network import get_network
from evaluator import DynUNetEvaluator
from transforms import determine_normalization_param_from_crop
from task_params import data_loader_params, patch_size, task_name
from trainer import DynUNetTrainer


def setup_root_logger():
    logger = logging.root
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt=None)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def check_input_args(args):

    # check the prior path argmument
    if args.prior_path and os.path.isfile(args.prior_path):
        print(f"Found prior: {args.prior_path}")
    elif args.prior_path and not os.path.isfile(args.prior_path):
        raise Exception(f"Prior file not found: {args.prior_path}")
    else:
        print("No prior provided with --prior_path ...")


def train(args):
    check_input_args(args)

    # load hyper parameters
    task_id = args.task_id
    fold = args.fold
    model_folds_dir = os.path.join(args.model_folds_dir, "task" + task_id, "runs_{}_fold{}_{}".format(
        args.task_id, args.fold, args.expr_name))
    resume_latest_checkpoint = args.resume_latest_checkpoint
    interval = args.interval
    learning_rate = args.learning_rate
    max_epochs = args.max_epochs
    multi_gpu_flag = args.multi_gpu
    amp_flag = args.amp
    lr_decay_flag = args.lr_decay
    sw_batch_size = args.sw_batch_size
    tta_val = args.tta_val
    batch_dice = args.batch_dice
    window_mode = args.window_mode
    eval_overlap = args.eval_overlap
    local_rank = args.local_rank
    determinism_flag = args.determinism_flag
    determinism_seed = args.determinism_seed
    prior_path = args.prior_path
    checkpoint = args.checkpoint
    datalist_path = args.datalist_path

    if determinism_flag:
        set_determinism(seed=determinism_seed)
        if local_rank == 0:
            print("Using deterministic training.")

    # set up the data loaders
    train_batch_size = data_loader_params[task_id]["batch_size"]
    if multi_gpu_flag:
        dist.init_process_group(backend="nccl", init_method="env://")

        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda")

    datalist_filepath = os.path.join(datalist_path, "dataset_task{}.json".format(task_id))
    properties = load_decathlon_properties(datalist_filepath, property_keys=["modality", "labels", "tensorImageSize"])

    # get datalists
    train_set_path = os.path.join(args.root_dir, task_name[task_id])
    datalist_train = get_datalist("train", datalist_path, task_id, properties['modality'], train_set_path=train_set_path, fold=fold, prior_path=prior_path)
    datalist_validation = get_datalist("validation", datalist_path, task_id, properties['modality'], train_set_path=train_set_path, fold=fold, prior_path=prior_path)

    # parameters used by transforms
    transform_params = {
        "pos_sample_num": args.pos_sample_num,
        "neg_sample_num": args.neg_sample_num,
        "use_prior": True if args.prior_path else False
    }

    # parameters used by dataloaders
    dataloader_params = {
        "train_num_workers": args.train_num_workers,
        "val_num_workers": args.val_num_workers,
    }

    # prep loader will perform pre-processing transforms including cropping and cache the dataset
    prep_loader = get_dataloader(datalist_train,
                                 transform_params,
                                 task_id,
                                 multi_gpu_flag,
                                 mode="prep",
                                 batch_size=train_batch_size,  # TODO: does batch_size have to be 1 for prep?
                                 **dataloader_params)

    # based on cached preprocessed dataset, additional transform parameters such as "use_nonzero" can be calculated
    transform_params["use_nonzero"] = determine_normalization_param_from_crop(prep_loader, key='image_0000')

    # subsequent data_loaders can continue from cached dataset
    val_loader = get_dataloader(datalist_validation,
                                transform_params,
                                task_id,
                                multi_gpu_flag,
                                mode="validation",
                                batch_size=1,
                                **dataloader_params)

    train_loader = get_dataloader(datalist_train,
                                  transform_params,
                                  task_id,
                                  multi_gpu_flag,
                                  mode="train",
                                  batch_size=train_batch_size,
                                  **dataloader_params)

    # produce the network (checkpoint is loaded later if provided)
    net = get_network(task_id,
                      n_classes=len(properties["labels"]),
                      n_in_channels=len(properties["modality"]),
                      prior_path=prior_path,
                      )
    net = net.to(device)

    if multi_gpu_flag:
        net = DistributedDataParallel(module=net, device_ids=[device])

    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=learning_rate,
        momentum=0.99,
        weight_decay=3e-5,
        nesterov=True,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / max_epochs) ** 0.9)

    # produce evaluator
    evaluator = DynUNetEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=net,
        num_classes=len(properties["labels"]),
        inferer=SlidingWindowInferer(
            roi_size=patch_size[task_id],
            sw_batch_size=sw_batch_size,
            overlap=eval_overlap,
            mode=window_mode,
        ),
        postprocessing=None,
        key_val_metric={
            "val_mean_dice": MeanDice(
                include_background=False,
                output_transform=from_engine(["pred", "label"]),
            )
        },
        val_handlers=None,
        amp=amp_flag,
        tta_val=tta_val,
    )

    # produce trainer
    loss = DiceCELoss(to_onehot_y=True, softmax=True, batch=batch_dice)
    trainer = DynUNetTrainer(
        device=device,
        max_epochs=max_epochs,
        train_data_loader=train_loader,
        network=net,
        optimizer=optimizer,
        loss_function=loss,
        inferer=SimpleInferer(),
        postprocessing=None,
        key_train_metric=None,
        train_handlers=None,
        amp=amp_flag,
    )

    # add evaluator handlers
    checkpoint_dict = {"net": net, "optimizer": optimizer, "scheduler": scheduler, "trainer": trainer}
    if idist.get_rank() == 0:
        checkpointSaver = CheckpointSaver(save_dir=model_folds_dir, save_dict=checkpoint_dict, save_key_metric=True)
        checkpointSaver.attach(evaluator)

    # add train handlers
    ValidationHandler(validator=evaluator, interval=interval, epoch_level=True).attach(trainer)
    if lr_decay_flag:
        lrScheduleHandler = LrScheduleHandler(lr_scheduler=scheduler, print_lr=True)
        lrScheduleHandler.attach(trainer)

    # print losses for each iteration only during first epoch
    def print_loss(engine):
        if idist.get_rank() == 0:
            if engine.state.epoch == 1:
                StatsHandler(
                    name="StatsHandler",
                    iteration_log=True,
                    epoch_log=False,
                    tag_name="train_loss",
                    output_transform=lambda x: x['loss'].item()
                ).iteration_completed(engine)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, print_loss)

    if local_rank > 0:
        evaluator.logger.setLevel(logging.WARNING)
        trainer.logger.setLevel(logging.WARNING)

    # store the training arguments in a json file for use during inference
    os.makedirs(model_folds_dir, exist_ok=True)
    with open(os.path.join(model_folds_dir, "training_params.json"), "w") as f:
        save_dict = vars(args)
        save_dict.update(transform_params)
        json.dump(save_dict, f, indent=2)

    # store the prior in model folder
    if prior_path:
        if os.path.isfile(prior_path):
            shutil.copy(prior_path, model_folds_dir)
        else:
            raise Exception(f"--prior_path provided but file not found: {properties['prior_path']}")

    # checkpoint handling
    if checkpoint:
        checkpoint_path = os.path.join(model_folds_dir, checkpoint)
        if os.path.exists(checkpoint_path):
            Checkpoint.load_objects(to_load=checkpoint_dict, checkpoint=checkpoint_path)
            print("Resuming from provided checkpoint: ", checkpoint_path)
        else:
            raise Exception(f"Provided checkpoint {checkpoint_path} not found.")
    elif resume_latest_checkpoint:
        checkpoints = glob(os.path.join(model_folds_dir, "checkpoint_key_metric=*.pt"))
        if len(checkpoints) == 0:
            print(f"No checkpoints found in {model_folds_dir}. Start training from beginning...")
        else:
            checkpoints.sort(key=lambda x: os.path.getmtime(x))  # sort by modification time
            checkpoint_latest = checkpoints[-1]  # pick the latest checkpoint
            print("Resuming from latest checkpoint: ", checkpoint_latest)
            Checkpoint.load_objects(to_load=checkpoint_dict, checkpoint=checkpoint_latest)

            # if max_epochs is provided as argument, it should overwrite the value stored in the trainer
            trainer.state.max_epochs = max_epochs
            # same for learning rate (note, this is the initial learning rate, not the learning rate at the resumed
            # epoch number)
            trainer.optimizer.param_groups[0]['initial_lr'] = learning_rate
            scheduler.base_lrs = [learning_rate for i in scheduler.base_lrs]

    else:
        print("No checkpoints provided. Start training from beginning...")

    trainer.run()
    if multi_gpu_flag:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-fold", "--fold", type=int, default=0, help="0-5")
    parser.add_argument("-task_id", "--task_id", type=str, default="04", help="task 01 to 10")
    parser.add_argument(
        "-root_dir",
        "--root_dir",
        type=str,
        default="/workspace/data/medical/",
        help="dataset path",
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
    )
    parser.add_argument(
        "-train_num_workers",
        "--train_num_workers",
        type=int,
        default=4,
        help="the num_workers parameter of training dataloader.",
    )
    parser.add_argument(
        "-val_num_workers",
        "--val_num_workers",
        type=int,
        default=1,
        help="the num_workers parameter of validation dataloader.",
    )
    parser.add_argument(
        "-interval",
        "--interval",
        type=int,
        default=5,
        help="the validation interval under epoch level.",
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
        default=4,
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
        "-pos_sample_num",
        "--pos_sample_num",
        type=int,
        default=2,
        help="the pos parameter of RandCropByPosNegLabeld.",
    )
    parser.add_argument(
        "-neg_sample_num",
        "--neg_sample_num",
        type=int,
        default=1,
        help="the neg parameter of RandCropByPosNegLabeld.",
    )
    parser.add_argument(
        "-cache_rate",
        "--cache_rate",
        type=float,
        default=1.0,
        help="the cache_rate parameter of CacheDataset.",
    )
    parser.add_argument("-learning_rate", "--learning_rate", type=float, default=1e-2)
    parser.add_argument(
        "-max_epochs",
        "--max_epochs",
        type=int,
        default=1000,
        help="number of epochs of training.",
    )
    parser.add_argument(
        "-model_folds_dir",
        "--model_folds_dir",
        type=str,
        default="",
        help="Path to folder that contains subfolders task01, task02... under which to store trained models",
    )
    parser.add_argument(
        "-checkpoint",
        "--checkpoint",
        type=str,
        default=None,
        help="the filename of weights.",
    )

    parser.add_argument('-resume_latest_checkpoint', '--resume_latest_checkpoint', dest='resume_latest_checkpoint', action='store_true',
                        help="whether to resume training from the latest modified checkpoint.")
    parser.set_defaults(resume_latest_checkpoint=False)

    parser.add_argument('-amp', '--amp', dest='amp', action='store_true', help="whether to use automatic mixed precision.")
    parser.add_argument('-no-amp', '--no-amp', dest='amp', action='store_false')
    parser.set_defaults(amp=True)

    parser.add_argument('-lr_decay', '--lr_decay', dest='lr_decay', action='store_true', help="whether to use learning rate decay.")
    parser.add_argument('-no-lr_decay', '--no-lr_decay', dest='lr_decay', action='store_false')
    parser.set_defaults(lr_decay=True)

    parser.add_argument('-tta_val', '--tta_val', dest='tta_val', action='store_true', help="whether to use test time augmentation.")
    parser.add_argument('-no-tta_val', '--no-tta_val', dest='tta_val', action='store_false')
    parser.set_defaults(tta_val=True)

    parser.add_argument('-batch_dice', '--batch_dice', dest='batch_dice', action='store_true', help="the batch parameter of Loss.")
    parser.add_argument('-no-batch_dice', '--no-batch_dice', dest='batch_dice', action='store_false')
    parser.set_defaults(batch_dice=False)

    parser.add_argument('-determinism_flag', '--determinism_flag', dest='determinism_flag', action='store_true')
    parser.add_argument('-no-determinism_flag', '--no-determinism_flag', dest='determinism_flag', action='store_false')
    parser.set_defaults(determinism_flag=False)

    parser.add_argument(
        "-determinism_seed",
        "--determinism_seed",
        type=int,
        default=0,
        help="the seed used in deterministic training",
    )

    parser.add_argument('-multi_gpu', '--multi_gpu', dest='multi_gpu', action='store_true', help="whether to use multiple GPUs for training.")
    parser.add_argument('-no-multi_gpu', '--no-multi_gpu', dest='multi_gpu', action='store_false')
    parser.set_defaults(multi_gpu=False)

    parser.add_argument(
        "-prior_path",
        "--prior_path",
        type=str,
        default="",
        help="prior passed as additional input channel to network, has to have same shape as input images and requires "
             "alignment between all input images and the prior",
    )

    parser.add_argument("-local_rank", "--local_rank", type=int, default=0)
    setup_root_logger()
    args = parser.parse_args()
    if args.local_rank == 0:
        print_config()
    train(args)
