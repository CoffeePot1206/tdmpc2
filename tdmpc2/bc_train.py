import datetime
import json
import shutil
import numpy as np
import time
import os
from omegaconf import OmegaConf
import psutil
import sys
import traceback
import h5py

from collections import OrderedDict
import hydra
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from robomimic.utils.log_utils import PrintLogger
import wandb

os.environ['MUJOCO_GL'] = 'egl'
from bc.model import ActorModel
from bc.dataset import BCDataset
from bc.rollout import rollout
from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env


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

    model_dir = os.path.join(base_output_dir, time_str, cfg.obs_type, "models")
    os.makedirs(model_dir)

    # tensorboard directory
    log_dir = os.path.join(base_output_dir, time_str, cfg.obs_type, "logs")
    os.makedirs(log_dir)

    # video directory
    video_dir = os.path.join(base_output_dir, time_str, cfg.obs_type, "videos")
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

    print("")

    # setup for a new training run
    data_logger = wandb
    data_logger.init(
        project="tdmpc2",
        entity=cfg.wandb_entity,
        name=cfg.wandb_name+"-"+cfg.obs_type,
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

        # Evaluate the model by by running rollouts

        # do rollouts at fixed rate or if it's time to save a new ckpt
        video_path = None
        rollout_check = (epoch % cfg.eval_freq == 0)
        if (epoch > rollout_start) and rollout_check:

            num_episodes = cfg.eval_episodes
            all_rollout_logs, video_path = rollout(
                model=model,
                env=env,
                horizon=cfg.eval_horizon,
                num_episodes=cfg.eval_episodes,
                video_dir=video_dir,
                epoch=epoch,
            )

            # summarize results from rollouts to tensorboard and terminal
            for k, v in all_rollout_logs.items():
                data_logger.log({"Rollout/{}".format(k): v}, epoch)

            print(json.dumps(all_rollout_logs, sort_keys=True, indent=4))

        # Save model checkpoints based on conditions (success rate, validation loss, etc)
        if should_save_ckpt:  # EDITED
            torch.save(model.net.state_dict(), f=os.path.join(ckpt_dir, "epoch_{}_success_{}.pth".format(epoch, all_rollout_logs["Success_Rate"])))

        # Finally, log memory usage in MB
        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss / 1000000)
        data_logger.log({"System/RAM Usage (MB)": mem_usage}, epoch)
        print("\nEpoch {} Memory Usage: {} MB\n".format(epoch, mem_usage))

    # # terminate logging
    # data_logger.close()


@hydra.main(config_name='cup_debug', config_path='./config')
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
        cfg.eval_horizon = 50
        cfg.start_from = 0

        # send output to a temporary directory
        cfg.work_dir = "/tmp/tmp_trained_models"

        # cfg.wandb_enabled = False

    # catch error during training and print it
    res_str = "finished run successfully!"
    try:
        train(cfg)
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    print(res_str)


if __name__ == "__main__":
    main()