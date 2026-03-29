"""
A script to convert a directory of .npz files into the LeRobot dataset format.

This script encapsulates data lookup and processing logic into separate functions,
making the code cleaner and easier to maintain.

Data processing logic:
- `observation.state`: end-effector pose and gripper width (7D), xyz(3)+euler(3)+gripper(1)
- `actions`: delta end-effector pose and gripper action (7D), delta_xyz(3)+delta_euler(3)+gripper(1)

Usage:
uv run examples/franka/convert_npy_to_lerobot.py --repo-id "pancake-w/test_npy" --data-dir "/nvme_data/bingwen/share_datasets/franka_panda/pick_to_plate-real"
"""

import os
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1" # disable progress bars for huggingface datasets

import shutil
import tyro
import numpy as np
import glob
from tqdm import tqdm
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME
from typing import Generator, Tuple, Dict, Any, Optional, List
from transforms3d.euler import quat2euler

def find_episode_files(data_dir: str) -> List[str]:
    """
    Find all 'data.npz' files for episodes in the specified directory.

    Args:
        data_dir: The parent directory containing 'episode_*' folders.

    Returns:
        A sorted list containing the full paths to all 'data.npz' files.
    """
    print(f"Searching for episodes in directory '{data_dir}'...")
    episode_files = sorted(glob.glob(os.path.expanduser(os.path.join(data_dir, "episode_*", "data.npz"))))
    
    if not episode_files:
        print(f"Warning: No files matching the 'episode_*/data.npz' pattern were found in '{data_dir}'.")
    else:
        print(f"Found {len(episode_files)} episode files.")
        
    return episode_files

def process_episode(npy_path: str, instruction: str) -> Optional[Tuple[str, int, Generator[Dict[str, Any], None, None]]]:
    """
    Load and process a single episode's .npz file.

    This function returns a generator that yields processed data frame-by-frame (step)
    to save memory.

    Args:
        npy_path: The path to a single 'data.npz' file.
        instruction: The task instruction for this episode.

    Returns:
        A tuple (instruction, num_steps, frame_generator), 
        or None if the episode is invalid (e.g., cannot be loaded or is empty).
    """
    try:
        episode_data = np.load(npy_path, allow_pickle=True)['data'].item()
    except Exception as e:
        tqdm.write(f"Error: Failed to load {npy_path}. Error message: {e}")
        return None

    # num_steps = len(episode_data['observation']['rgb'])
    len_obs = len(episode_data['observation']['rgb'])
    len_state = len(episode_data['state']['joint']['position'])
    len_action = len(episode_data['action']['end_effector']['delta_position'])
    
    num_steps = min(len_obs, len_state, len_action)
    if num_steps == 0:
        tqdm.write(f"Warning: {os.path.dirname(npy_path)} is an empty episode, skipping.")
        return None
    
    def frame_generator():
        """A generator that yields data frame by frame."""
        for step_idx in range(num_steps):
            # ================= 1. State (修改部分) =================
            # 获取前7维 qpos
            state_qpos = episode_data['state']['joint']['position'][step_idx]
            
            # 获取 gripper_width，除以2，并复制为两维
            gripper_width = episode_data['state']['end_effector']['gripper_width'][step_idx]
            half_width = gripper_width / 2.0
            state_gripper = np.array([half_width, half_width])
            
            # 拼接成 9 维向量 (7维 qpos + 2维 gripper)
            state_vec = np.concatenate([state_qpos, state_gripper]).astype(np.float32)

            # ================= 2. Action (保持不变) =================
            ee_action_data = episode_data['action']['end_effector']
            action_delta_pos = ee_action_data['delta_position'][step_idx]
            action_delta_ori = ee_action_data['delta_orientation'][step_idx]
            action_delta_euler = quat2euler(action_delta_ori, axes='sxyz') # Convert quaternion to Euler angles
            action_gripper = np.array([ee_action_data['gripper_control'][step_idx]]) # [0, 1]
            action_vec = np.concatenate([action_delta_pos, action_delta_euler, action_gripper]).astype(np.float32)

            # 注意：这里的维度校验需要将 state_vec.shape[0] 更新为 9
            if state_vec.shape[0] != 9 or action_vec.shape[0] != 7:
                # tqdm.write(f"Error: Dimension mismatch at step {step_idx} in {os.path.dirname(npy_path)}. Skipping this frame.")
                continue
                
            yield {
                "image": episode_data['observation']['rgb'][step_idx, 1],
                "state": state_vec,     # 维度: 9 (7 qpos + 2 gripper)
                "actions": action_vec,  # 维度: 7 (3 pos + 3 euler + 1 gripper)
            }

    # def frame_generator():
    #     """A generator that yields data frame by frame."""
    #     for step_idx in range(num_steps):
    #         # State
    #         ee_state_data = episode_data['state']['end_effector']
    #         state_pos = ee_state_data['position'][step_idx]
    #         state_ori = ee_state_data['orientation'][step_idx]
    #         state_euler = quat2euler(state_ori, axes='sxyz')
    #         state_gripper = np.array([ee_state_data['gripper_width'][step_idx]])
    #         state_vec = np.concatenate([state_pos, state_euler, state_gripper]).astype(np.float32)

    #         # Action
    #         ee_action_data = episode_data['action']['end_effector']
    #         action_delta_pos = ee_action_data['delta_position'][step_idx]
    #         action_delta_ori = ee_action_data['delta_orientation'][step_idx]
    #         action_delta_euler = quat2euler(action_delta_ori, axes='sxyz') # Convert quaternion to Euler angles
    #         action_gripper = np.array([ee_action_data['gripper_control'][step_idx]]) # [0, 1]
    #         action_vec = np.concatenate([action_delta_pos, action_delta_euler, action_gripper]).astype(np.float32)

    #         if state_vec.shape[0] != 7 or action_vec.shape[0] != 7:
    #             # tqdm.write(f"Error: Dimension mismatch at step {step_idx} in {os.path.dirname(npy_path)}. Skipping this frame.")
    #             continue
                
    #         yield {
    #             "image": episode_data['observation']['rgb'][step_idx, 1],
    #             "state": state_vec, # 7
    #             "actions": action_vec, # 7
    #         }

    return instruction, num_steps, frame_generator()

def main(repo_id: str, data_dir: str, instruction: str, *, push_to_hub: bool = False, max_episode_num: Optional[int] = None):
    """
    The main conversion function that orchestrates the entire workflow.
    """
    # --- 1. Set up output path and clean up old data ---
    # output_path = HF_LEROBOT_HOME / repo_id
    output_path = f"datasets/{repo_id}"
    if os.path.exists(output_path):
        print(f"Removing existing dataset: {output_path}")
        shutil.rmtree(output_path)

    # --- 2. Define and create the LeRobot dataset structure ---
    print("Creating LeRobot dataset structure...")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_path,
        robot_type="franka",
        fps=10,
        features={
            "observation.image": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32", 
                "shape": (9,),
            },
            "actions": {
                "dtype": "float32", 
                "shape": (7,), 
            },
        },
        use_videos=True,
        image_writer_threads=4,
        image_writer_processes=2,
    )

    # --- 3. Find all episode files ---
    episode_files = find_episode_files(data_dir)
    if not episode_files:
        print("No valid data found, exiting program.")
        return
    
    # Limit number of episodes if specified
    if max_episode_num is not None:
        episode_files = episode_files[:max_episode_num]
        print(f"Processing only the first {len(episode_files)} episodes (max_episode_num={max_episode_num})")

    # --- 4. Iterate, process, and write to the LeRobot dataset ---
    for idx, npy_path in enumerate(tqdm(episode_files, desc="Processing Episodes"), 1):
        processed_data = process_episode(npy_path, instruction)
        if processed_data is None:
            continue

        instruction, num_steps, frame_generator = processed_data        
        
        # Add data frame by frame aligned to the new format
        for frame_data in frame_generator:
            dataset.add_frame({
                "observation.image": frame_data["image"],
                "observation.state": frame_data["state"],
                "actions": frame_data["actions"],
                "task": instruction,
            })

        dataset.save_episode()
        tqdm.write(f"[{idx}/{len(episode_files)}] Saved episode for task '{instruction}' (with {num_steps} steps)")

    if push_to_hub:
        print("Pushing to Hugging Face Hub...")
        dataset.push_to_hub(
            tags=["franka", "pick-and-place", "robotics", "openpi"],
            private=True,
            push_videos=True,
            license="apache-2.0",
        )

    print("\nConversion complete!")
    print(f"LeRobot dataset saved to: {output_path}")

if __name__ == "__main__":
    tyro.cli(main)