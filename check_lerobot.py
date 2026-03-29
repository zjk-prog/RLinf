import os
import tyro
import torch
import numpy as np
import matplotlib.pyplot as plt
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def check_first_frame(repo_id: str, episode_idx: int = 0):
    """
    读取 LeRobot 数据集中指定 Episode 的起始帧并进行可视化检查。
    """
    dataset_dir = f"datasets/{repo_id}"
    
    if not os.path.exists(dataset_dir):
        print(f"Error: 找不到数据集目录 {dataset_dir}")
        return

    print(f"Loading LeRobot dataset from {dataset_dir}...")
    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_dir)
    
    if episode_idx >= dataset.num_episodes:
        print(f"Error: 指定的 episode_idx {episode_idx} 超出范围 (总数 {dataset.num_episodes})")
        return

    # 获取指定 episode 的起始帧索引
    ep_from = dataset.episode_data_index["from"][episode_idx].item()
    
    # 仅读取起始帧
    frame = dataset[ep_from]
    
    # 提取数据
    img_tensor = frame["observation.image"]
    state = frame["observation.state"].numpy()
    action = frame["actions"].numpy()
    task = frame.get("task", "Unknown Task")

    # 打印读取到的原始形状，验证 LeRobot 的输出格式
    print(f"\n--- 第 {episode_idx} 个 Episode 的起始帧信息 ---")
    print(f"Task: {task}")
    print(f"Image Type: {type(img_tensor)}")
    print(f"Image Shape (From Dataset): {img_tensor.shape}  <-- 注意这里通常是 (C, H, W)")
    print(f"State Shape: {state.shape}")
    print(f"Action Shape: {action.shape}")
    
    print(f"\n[State Values] pos, euler, grp:\n{state}")
    print(f"[Action Values] delta_pos, delta_euler, grp:\n{action}")

    # 处理图像用于 matplotlib 显示 (需转换回 H, W, C)
    if isinstance(img_tensor, torch.Tensor):
        img_np = img_tensor.permute(1, 2, 0).numpy()
    else:
        img_np = img_tensor
        
    if img_np.max() > 1.0:
        img_np = img_np / 255.0

    # 可视化
    plt.figure(figsize=(6, 6))
    plt.imshow(img_np)
    plt.title(f"Episode {episode_idx} - Start Frame\nTask: {task}")
    plt.axis("off")
    
    info_text = (
        f"State: {np.round(state, 3)}\n"
        f"Action: {np.round(action, 3)}"
    )
    plt.text(0.5, -0.05, info_text, transform=plt.gca().transAxes, 
             fontsize=10, ha='center', va='top', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2) 
    
    save_path = f"episode_{episode_idx}_start_frame.png"
    plt.savefig(save_path, dpi=150)
    print(f"\n可视化图像已保存至: {save_path}")
    plt.show()

    # 如果 state 是 9 维的，寻找 mid-frame
    if state.shape[0] == 9:
        print("\n--- 检测到 9 维 State，开始根据 Action 最后一维寻找 Mid-Frame ---")
        ep_to = dataset.episode_data_index["to"][episode_idx].item()
        
        # 提取整个 episode 的 gripper action (最后一维)
        episode_length = ep_to - ep_from
        mid_idx_relative = episode_length // 2  # 默认中间帧
        
        # 寻找夹爪动作发生变化的时刻
        prev_gripper = dataset[ep_from]["actions"].numpy()[-1]
        for i in range(1, episode_length):
            curr_gripper = dataset[ep_from + i]["actions"].numpy()[-1]
            if abs(curr_gripper - prev_gripper) > 0.01:
                mid_idx_relative = i
                break
                
        # 输出连续 4 帧
        for offset in range(8):
            curr_idx_relative = mid_idx_relative + offset
            curr_idx = ep_from + curr_idx_relative
            
            if curr_idx >= ep_to:
                print(f"\n已到达 episode 末尾，结束输出。")
                break

            curr_frame = dataset[curr_idx]
            
            mid_img_tensor = curr_frame["observation.image"]
            mid_state = curr_frame["observation.state"].numpy()
            mid_action = curr_frame["actions"].numpy()
            
            print(f"\n--- 第 {episode_idx} 个 Episode 的 Mid-Frame (Step {curr_idx_relative}) 信息 ---")
            print(f"Task: {task}")
            print(f"Image Type: {type(mid_img_tensor)}")
            print(f"Image Shape: {mid_img_tensor.shape}")
            print(f"State Shape: {mid_state.shape}")
            print(f"Action Shape: {mid_action.shape}")
            print(f"\n[State Values] pos, euler, grp:\n{mid_state}")
            print(f"[Action Values] delta_pos, delta_euler, grp:\n{mid_action}")

            # 处理图像用于 matplotlib 显示
            if isinstance(mid_img_tensor, torch.Tensor):
                mid_img_np = mid_img_tensor.permute(1, 2, 0).numpy()
            else:
                mid_img_np = mid_img_tensor
                
            if mid_img_np.max() > 1.0:
                mid_img_np = mid_img_np / 255.0

            plt.figure(figsize=(6, 6))
            plt.imshow(mid_img_np)
            plt.title(f"Episode {episode_idx} - Mid Frame (Step {curr_idx_relative})\nTask: {task}")
            plt.axis("off")
            
            mid_info_text = (
                f"State: {np.round(mid_state, 3)}\n"
                f"Action: {np.round(mid_action, 3)}"
            )
            plt.text(0.5, -0.05, mid_info_text, transform=plt.gca().transAxes, 
                     fontsize=10, ha='center', va='top', 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2) 
            
            mid_save_path = f"episode_{episode_idx}_mid_frame_plus_{offset}.png"
            plt.savefig(mid_save_path, dpi=150)
        print(f"\n可视化 Mid-Frame 图像已保存至: {mid_save_path}")
        plt.show()

if __name__ == "__main__":
    tyro.cli(check_first_frame)