
import argparse
import datetime
import json
import shutil
import numpy as np
import time
import os
from omegaconf import OmegaConf
import psutil
import sys
# sys.path.append("/home/kuangfuhang/tdmpc2/tdmpc2/tdmpc2")
import traceback
import h5py

from collections import OrderedDict
import hydra
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import robomimic
import robomimic.utils.train_utils as TrainUtils
from robomimic.utils.log_utils import PrintLogger
import wandb

from bc.model import ActorModel
from bc.dataset import BCDataset
from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2

def get_work_dir(cfg):
    t_now = time.time()
    time_str = datetime.datetime.fromtimestamp(t_now).strftime('%Y%m%d%H%M%S')
    
    base_output_dir = os.path.expanduser(cfg.work_dir)
    base_output_dir = os.path.join(base_output_dir, cfg.task)
    if os.path.exists(base_output_dir):
        ans = input("WARNING: model directory ({}) already exists! \noverwrite? (y/n)\n".format(base_output_dir))
        if ans == "y":
            print("REMOVING")
            shutil.rmtree(base_output_dir)

    model_dir = os.path.join(base_output_dir, time_str, "models")
    os.makedirs(model_dir)

    # tensorboard directory
    log_dir = os.path.join(base_output_dir, time_str, "logs")
    os.makedirs(log_dir)

    # video directory
    video_dir = os.path.join(base_output_dir, time_str, "videos")
    os.makedirs(video_dir)

    return model_dir, log_dir, video_dir

def run_epoch(model, data_loader, epoch, validate=False, num_steps=None):
    
    if num_steps is None:
        num_steps = len(data_loader)

    step_log_all = []

    data_loader_iter = iter(data_loader)
    for _ in tqdm(range(num_steps)):

        # load next batch from data loader
        try:
            batch = next(data_loader_iter)
        except StopIteration:
            # reset for next dataset pass
            data_loader_iter = iter(data_loader)
            batch = next(data_loader_iter)

        # forward and backward pass
        info = model.run_step(batch, validate=validate)

        # tensorboard logging
        step_log_all.append(info)

    # flatten and take the mean of the metrics
    step_log_dict = {}
    for i in range(len(step_log_all)):
        for k in step_log_all[i]:
            if k not in step_log_dict:
                step_log_dict[k] = []
            step_log_dict[k].append(step_log_all[i][k])
    step_log_all = dict((k, float(np.mean(v))) for k, v in step_log_dict.items())

    return step_log_all

def train(cfg):
    """
    Train a model using the algorithm.
    """

    # to make result reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    
    assert torch.cuda.is_available()
    assert cfg.eval_episodes > 0, 'Must evaluate at least 1 episode.'

    # Set seed and make environment
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    
    env = make_env(cfg)

    torch.set_num_threads(2)

    print("\n============= New Training Run with Config =============")
    print(cfg)
    print("")
    ckpt_dir, log_dir, video_dir = get_work_dir(cfg)

    # log stdout and stderr to a text file
    logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
    sys.stdout = logger
    sys.stderr = logger

    # make sure the dataset exists
    dataset_path = os.path.expanduser(cfg.dataset)
    if not os.path.exists(dataset_path):
        raise Exception("Dataset at provided path {} not found!".format(dataset_path))
    

    env_meta = {}

    shape_meta = {
        'ac_dim': 2,
        'all_shapes': OrderedDict([
            ('rgb', [3, 84, 84]),
            ('position', [4]),
            ('velocity', [4]),
        ]),
        'all_obs_keys': [
            'rgb',
            'position',
            'velocity',
        ],
        'use_images': True,
        'use_depths': False,
	}

    # env = get_env()
    # print(env)
    # exit()
    envs = {"cup-catch": env}

    print("")

    # setup for a new training run
    data_logger = wandb
    data_logger.init(
        project="tdmpc2",
        entity=cfg.wandb_entity,
        name=cfg.wandb_name,
        config=OmegaConf.to_container(cfg=cfg),
        mode="online" if cfg.wandb_enabled else "disabled"
    )
    
    # load model
    model = ActorModel(
        cfg=cfg,
        # obs_shapes=shape_meta["all_shapes"],
        obs_shapes=cfg.obs_shapes,
        ac_dim=shape_meta["ac_dim"],
        mlp_layer_dims=[512, 512]
    )
    
    # save the config as a json file
    with open(os.path.join(log_dir, '..', 'config.yaml'), 'w') as outfile:
        OmegaConf.save(config=cfg, f=outfile)

    print("\n============= Model Summary =============")
    print(model)  # print model summary
    print("")

    # load training data    # TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    trainset, validset = BCDataset(cfg.dataset, split='train'), BCDataset(cfg.dataset, split='valid')
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")
    if validset is not None:
        print("\n============= Validation Dataset =============")
        print(validset)
        print("")

    # initialize data loaders
    train_loader = DataLoader(
        dataset=trainset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        drop_last=True,
        # pin_memory=torch.cuda.is_available()
    )

    if cfg.validate:
        # cap num workers for validation dataset at 1
        num_workers = min(cfg.workers, 1)
        valid_loader = DataLoader(
            dataset=validset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            # pin_memory=torch.cuda.is_available()
        )
    else:
        valid_loader = None

    # main training loop
    best_valid_loss = None
    best_return = -np.inf
    best_success_rate = -1

    # number of learning steps per epoch (defaults to a full dataset pass)
    train_num_steps = cfg.steps_per_epoch
    valid_num_steps = cfg.valid_steps_per_epoch

    for epoch in range(1, cfg.num_epochs + 1): # epoch numbers start at 1
        step_log = run_epoch(
            model=model,
            data_loader=train_loader,
            epoch=epoch,
            num_steps=train_num_steps,
        )
        model.on_epoch_end(epoch)

        # setup checkpoint path
        epoch_ckpt_name = "model_epoch_{}".format(epoch)

        # check for recurring checkpoint saving conditions
        should_save_ckpt = False
        rollout_start = cfg.start_from
        # rollout the last 5 ckpt by default
        if rollout_start is None:
            rollout_start = cfg.num_epochs - 5 * cfg.eval_freq
        if cfg.save_model:
            should_save_ckpt = ((epoch % cfg.eval_freq) == 0) and (epoch > rollout_start)

        print("Train Epoch {}".format(epoch))
        print(json.dumps(step_log, sort_keys=True, indent=4))
        for k, v in step_log.items():
            data_logger.log({"Train/{}".format(k): v}, epoch)

        # Evaluate the model on validation set
        if cfg.validate:
            with torch.no_grad():
                step_log = run_epoch(model=model, data_loader=valid_loader, epoch=epoch, validate=True, num_steps=valid_num_steps)
            for k, v in step_log.items():
                data_logger.log({"Valid/{}".format(k): v}, epoch)

            print("Validation Epoch {}".format(epoch))
            print(json.dumps(step_log, sort_keys=True, indent=4))

            # save checkpoint if achieve new best validation loss
            valid_check = "Loss" in step_log
            if valid_check and (best_valid_loss is None or (step_log["Loss"] <= best_valid_loss)):
                best_valid_loss = step_log["Loss"]
                if cfg.experiment.save.enabled and cfg.experiment.save.on_best_validation:
                    epoch_ckpt_name += "_best_validation_{}".format(best_valid_loss)
                    should_save_ckpt = True
                    ckpt_reason = "valid" if ckpt_reason is None else ckpt_reason

        continue 
        # Evaluate the model by by running rollouts

        # do rollouts at fixed rate or if it's time to save a new ckpt
        video_paths = None
        rollout_check = (epoch % cfg.eval_freq == 0)
        if (epoch > rollout_start) and rollout_check:

            # wrap model as a RolloutPolicy to prepare for rollouts
            rollout_model = RolloutPolicy(model, obs_normalization_stats=obs_normalization_stats)

            num_episodes = cfg.experiment.rollout.n
            all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
                policy=rollout_model,
                envs=envs,
                horizon=cfg.experiment.rollout.horizon,
                use_goals=cfg.use_goals,
                num_episodes=num_episodes,
                render=False,
                video_dir=video_dir if cfg.save_video else None,
                epoch=epoch,
                video_skip=cfg.experiment.get("video_skip", 5),
                terminate_on_success=cfg.experiment.rollout.terminate_on_success,
            )

            # summarize results from rollouts to tensorboard and terminal
            for env_name in all_rollout_logs:
                rollout_logs = all_rollout_logs[env_name]
                for k, v in rollout_logs.items():
                    data_logger.log({"Rollout/{}/{}".format(k, env_name): v}, epoch, log_stats=True)

                print("\nEpoch {} Rollouts took {}s (avg) with results:".format(epoch, rollout_logs["time"]))
                print('Env: {}'.format(env_name))
                print(json.dumps(rollout_logs, sort_keys=True, indent=4))

            # checkpoint and video saving logic
            updated_stats = TrainUtils.should_save_from_rollout_logs(
                all_rollout_logs=all_rollout_logs,
                best_return=best_return,
                best_success_rate=best_success_rate,
                epoch_ckpt_name=epoch_ckpt_name,
                save_on_best_rollout_return=cfg.experiment.save.on_best_rollout_return,
                save_on_best_rollout_success_rate=cfg.experiment.save.on_best_rollout_success_rate,
            )
            best_return = updated_stats["best_return"]
            best_success_rate = updated_stats["best_success_rate"]
            epoch_ckpt_name = updated_stats["epoch_ckpt_name"]
            should_save_ckpt = (cfg.experiment.save.enabled and updated_stats["should_save_ckpt"]) or should_save_ckpt
            if updated_stats["ckpt_reason"] is not None:
                ckpt_reason = updated_stats["ckpt_reason"]

        # Only keep saved videos if the ckpt should be saved (but not because of validation score)
        should_save_video = (should_save_ckpt and (ckpt_reason != "valid")) or cfg.experiment.keep_all_videos
        if video_paths is not None and not should_save_video:
            for env_name in video_paths:
                os.remove(video_paths[env_name])

        # Save model checkpoints based on conditions (success rate, validation loss, etc)
        if should_save_ckpt:  # EDITED
            TrainUtils.save_model(
                model=model,
                config=cfg,
                env_meta=env_meta,
                shape_meta=shape_meta,
                ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                obs_normalization_stats=obs_normalization_stats,
            )

        # Finally, log memory usage in MB
        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss / 1000000)
        data_logger.log("System/RAM Usage (MB)", mem_usage, epoch)
        print("\nEpoch {} Memory Usage: {} MB\n".format(epoch, mem_usage))

    # # terminate logging
    # data_logger.close()


@hydra.main(config_name='cup_train', config_path='./config')
def main(cfg: dict):
    
    # maybe modify config for debugging purposes
    if cfg.debug:

        # train and validate (if enabled) for 3 gradient steps, for 2 epochs
        cfg.steps_per_epoch = 3
        cfg.valid_steps_per_epoch = 3
        cfg.num_epochs = 2

        # if rollouts are enabled, try 2 rollouts at end of each epoch, with 10 environment steps
        cfg.eval_episodes = 2
        cfg.eval_freq = 2
        cfg.eval_horizon = 10

        # send output to a temporary directory
        cfg.work_dir = "/tmp/tmp_trained_models"

        cfg.wandb_enabled = False

    # catch error during training and print it
    res_str = "finished run successfully!"
    try:
        train(cfg)
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    print(res_str)


if __name__ == "__main__":
    main()