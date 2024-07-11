import json
import os
import numpy as np
import imageio
from tqdm import tqdm
import cv2
import torch

def rollout(
    model, 
    env,
    horizon,
    num_episodes,
    video_dir,
    epoch,
    video_skip=1,
    terminate_on_success=True,
):
    video_name = "epoch_{}.mp4".format(epoch)
    video_path = os.path.join(video_dir, video_name)
    video_writer = imageio.get_writer(video_path, fps=20)

    rollout_logs = []

    num_success = 0
    for ep in tqdm(range(num_episodes)):
        rollout_info = rollout_once(
            model=model,
            env=env,
            horizon=horizon,
            video_writer=video_writer,
            epoch=epoch,
            video_skip=video_skip,
            terminate_on_success=terminate_on_success,
            ep=ep,
        )

        rollout_logs.append(rollout_info)
        num_success += rollout_info["Success_Rate"]
        print("Episode {}, horizon={}, num_success={}".format(ep + 1, horizon, num_success))
        print(json.dumps(rollout_info, sort_keys=True, indent=4))

    video_writer.close()

    # average metric across all episodes
    rollout_logs = dict((k, [rollout_logs[i][k] for i in range(len(rollout_logs))]) for k in rollout_logs[0])
    rollout_logs_mean = dict((k, np.mean(v)) for k, v in rollout_logs.items())

    return rollout_logs_mean, video_path


def rollout_once(
    model, 
    env,
    horizon,
    video_writer,
    epoch,
    ep,
    video_skip=1,
    terminate_on_success=True,
):
    obs = dict()
    results = dict()

    state = env.reset(task_idx=0)
    rgb = env.render(
        mode='rgb_array', width=84, height=84
    )

    obs['rgb'] = torch.from_numpy(rgb.copy())
    obs['position'] = state[:4]
    obs['velocity'] = state[4:]

    success = False
    video_count = 0
    total_reward = 0.

    for i in range(horizon):
        action = model.get_action(obs)

        state, reward, done, t = env.step(action)

        rgb = env.render(
            mode='rgb_array', width=84, height=84
        )
        obs['rgb'] = torch.from_numpy(rgb.copy())
        obs['position'] = state[:4]
        obs['velocity'] = state[4:]

        total_reward += reward
        success = (reward >= 2)
        
        if video_count % video_skip == 0:
            video_img = env.render(mode="rgb_array", height=256, width=256)
            video_img = cv2.putText(video_img.copy(), text="{}".format(ep+1), org=(5, 20), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(255, 255, 255))
            video_writer.append_data(video_img)
        video_count += 1

        if terminate_on_success and success:
            break

    results["Reward"] = total_reward.item()
    results["Horizon"] = i + 1
    results["Success_Rate"] = float(success)

    return results