from collections import deque
import os

from tqdm import tqdm
os.environ['MUJOCO_GL'] = 'egl'
import warnings
warnings.filterwarnings('ignore')

import hydra
import imageio
import numpy as np
import torch
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2

import h5py

torch.backends.cudnn.benchmark = True


def generate_traj(
	env,
	agent,
	task_idx,
	cfg,
	maxlen=1,
):
	obs = {
		"rgb": [],
		"position": [],
		"velocity": [],
	}
	actions = []
	rewards = []
	dones = []
	frames = deque([], maxlen=maxlen)
	success = False
	video = []
	if cfg.save_video:
		video = [env.render()]

	state, done, reward, t = env.reset(task_idx=task_idx), False, 0, 0

	while not done:
		action = agent.act(state, t0=t==0, task=task_idx)

		frame = env.render(
			mode='rgb_array', width=84, height=84
		)
		frames.append(frame)

		obs['rgb'].append(np.concatenate(frames))
		obs['position'].append(np.array(state[:4]))
		obs['velocity'].append(np.array(state[4:]))
		actions.append(np.array(action))
		rewards.append(np.array(reward))
		dones.append(np.array(done))

		if reward >= 2:
			success = True

		state, reward, done, info = env.step(action)

		if cfg.save_video:
			video.append(env.render())
		
		t += 1
	
	for k in obs:
		obs[k] = np.array(obs[k])
	actions = np.array(actions)
	rewards = np.array(rewards)
	dones = np.array(dones)
	
	traj = {
		"obs": obs,
		"actions": actions,
		"rewards": rewards,
		"dones": dones,
	}

	return traj, success, video


@hydra.main(config_name='cup_collect', config_path='./config')
def evaluate(cfg: dict):
	"""
	Script for evaluating a single-task / multi-task TD-MPC2 checkpoint.

	Most relevant args:
		`task`: task name (or mt30/mt80 for multi-task evaluation)
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`checkpoint`: path to model checkpoint to load
		`eval_episodes`: number of episodes to evaluate on per task (default: 10)
		`save_video`: whether to save a video of the evaluation (default: True)
		`seed`: random seed (default: 1)
	
	See config.yaml for a full list of args.

	Example usage:
	````
		$ python evaluate.py task=mt80 model_size=48 checkpoint=/path/to/mt80-48M.pt
		$ python evaluate.py task=mt30 model_size=317 checkpoint=/path/to/mt30-317M.pt
		$ python evaluate.py task=dog-run checkpoint=/path/to/dog-1.pt save_video=true
	```
	"""
	assert torch.cuda.is_available()
	assert cfg.eval_episodes > 0, 'Must evaluate at least 1 episode.'
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	print(colored(f'Task: {cfg.task}', 'blue', attrs=['bold']))
	print(colored(f'Model size: {cfg.get("model_size", "default")}', 'blue', attrs=['bold']))
	print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))
	if not cfg.multitask and ('mt80' in cfg.checkpoint or 'mt30' in cfg.checkpoint):
		print(colored('Warning: single-task evaluation of multi-task models is not currently supported.', 'red', attrs=['bold']))
		print(colored('To evaluate a multi-task model, use task=mt80 or task=mt30.', 'red', attrs=['bold']))

	# Make environment
	env = make_env(cfg)

	# Load agent
	agent = TDMPC2(cfg)
	assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found! Must be a valid filepath.'
	agent.load(cfg.checkpoint)

	# create dataset
	output_path = "/cache1/kuangfuhang/tdmpc2/datasets/cup-catch/rgbd-3.hdf5"
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	f_out = h5py.File(output_path, "w")
	data_grp = f_out.create_group("data")
	mask_grp = f_out.create_group("mask")

	total_samples = 0
	added_demos = []
	
	# Evaluate
	if cfg.multitask:
		print(colored(f'Evaluating agent on {len(cfg.tasks)} tasks:', 'yellow', attrs=['bold']))
	else:
		print(colored(f'Evaluating agent on {cfg.task}:', 'yellow', attrs=['bold']))
	if cfg.save_video:
		video_dir = os.path.join(cfg.work_dir, 'videos')
		os.makedirs(video_dir, exist_ok=True)
	scores = []
	tasks = cfg.tasks if cfg.multitask else [cfg.task]
	for task_idx, task in enumerate(tasks):
		if not cfg.multitask:
			task_idx = None
		ep_rewards, ep_successes = [], []
		ep = 0
		while ep < cfg.eval_episodes:
			traj, success, frames = generate_traj(
						env,
						agent,
						task_idx,
						cfg,
					)
			if not success:
				continue
			added_demos.append(f"demo_{ep}")
			# ep_rewards.append(ep_reward)
			# ep_successes.append(info['success'])
			if cfg.save_video:
				imageio.mimsave(
					os.path.join(video_dir, f'{task}-{ep}.mp4'), frames, fps=15)
			ep_data_grp = data_grp.create_group(f"demo_{ep}")
			ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
			# ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
			ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
			ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
			for k in traj["obs"]:
				ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
				# if not args.exclude_next_obs:
				# 	ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

			ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0]
			total_samples += traj["actions"].shape[0]

			print(f"ep: {ep} done")
			ep += 1
		

    # global metadata
	data_grp.attrs["total"] = total_samples
	
    # generate filter keys
	np.random.seed(1)
	np.random.shuffle(added_demos)
	cut_off = len(added_demos) // 10 + 1
	valid_demos = added_demos[:cut_off]
	train_demos = added_demos[cut_off:]
	mask_grp.create_dataset("train", data=train_demos)
	mask_grp.create_dataset("valid", data=valid_demos)

	f_out.close()



if __name__ == '__main__':
	evaluate()
