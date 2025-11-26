"""Data collection script for RSL-RL trained policy."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# # add argparse arguments
parser = argparse.ArgumentParser(description="Collect RL data with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
# parser.add_argument(
#     "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
# )
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--num_demos", type=int, default=10, help="Number of demos to collect.")
# parser.add_argument(
#     "--use_pretrained_checkpoint",
#     action="store_true",
#     help="Use the pre-trained checkpoint from Nucleus.",
# )
# parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# # append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# # append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    # Override configs from CLI
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs or env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # Load checkpoint and runner
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        # finds the latest checkpoint in log_root_path
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # if isinstance(env.unwrapped, DirectMARLEnv):
    #     env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # Get policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Data buffers
    collected_data = {"observations": [], "actions": [], "rewards": [], "dones": []}

    obs = env.get_observations()
    
    total_steps = 0
    # max_steps = (
    #     args_cli.num_demos
    # )  # total number of steps to collect, or trajectories * episode length

    frames = []
    import numpy as np
    import cv2

    for _ in range(10):
        dones = np.zeros(env.num_envs, dtype=bool)
        while simulation_app.is_running() and not dones[0]:
            frames.append(env.env.render())
            with torch.inference_mode():
                actions = policy(obs)
                next_obs, rewards, dones, infos = env.step(actions)

            # Collect data
            collected_data["observations"].append(obs)
            collected_data["actions"].append(actions)
            collected_data["rewards"].append(rewards)
            collected_data["dones"].append(dones)

            obs = next_obs
            total_steps += 1

            # Reset envs where done is True if needed (usually handled by VecEnv wrapper)
            # Optionally, you can also save data periodically here

    frames = np.array(frames)
    h, w = frames.shape[1:3]
    print(frames.shape)
    video_path = 'out.mp4'
    # Remove the video file if it exists, to ensure overwriting
    if os.path.exists(video_path):
        os.remove(video_path)
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
    for f in frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()

    import ipdb; ipdb.set_trace()

    # Save collected data
    import pickle

    save_path = args_cli.save_path or "collected_rl_data.pkl"
    print(f"Saving collected data to {save_path}")
    with open(save_path, "wb") as f:
        pickle.dump(collected_data, f)

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
