import os
import shutil

# ==========================================
# 拔掉网线：强制开启离线模式 (必须在 import lerobot 之前)
# ==========================================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"

# 设置本地数据集库的根目录
BASE_DIR = "/workspace/RL-co-for-gs/datasets"
os.environ["HF_LEROBOT_HOME"] = BASE_DIR

# 环境变量设置完毕后，再导入相关的库
import numpy as np
from tqdm import tqdm
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def main():
    gs_repo = "GSEnv-pnp"
    real_repo = "real-pnp"
    mix_repo = "mix-pnp"
    
    out_path = os.path.join(BASE_DIR, mix_repo)
    
    print("🔌 已开启彻底离线模式，准备加载本地原始数据集...")
    # 这里因为上面设置了离线模式和 HOME 目录，它绝对不会再去报错连网了
    ds_gs = LeRobotDataset(repo_id=gs_repo, root=os.path.join(BASE_DIR, gs_repo))
    ds_real = LeRobotDataset(repo_id=real_repo, root=os.path.join(BASE_DIR, real_repo))
    
    if os.path.exists(out_path):
        print(f"🧹 检测到旧文件夹，正在清理: {out_path}")
        shutil.rmtree(out_path)
        
    ds_mix = LeRobotDataset.create(
        repo_id=mix_repo,
        root=out_path,
        robot_type="franka",      # 修改 1：直接指定 franka
        fps=10,                   # 修改 2：直接指定 fps 为 10
        features={                # 修改 3：直接把你之前的特征字典贴过来
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
    
    def copy_episodes(source_ds, prefix_name):
        from_indices = source_ds.episode_data_index["from"].numpy()
        to_indices = source_ds.episode_data_index["to"].numpy()
        num_episodes = len(from_indices)
        
        for ep_idx in tqdm(range(num_episodes), desc=f"合并 {prefix_name}"):
            start_idx = from_indices[ep_idx]
            end_idx = to_indices[ep_idx]
            
            hf_item = source_ds.hf_dataset[int(start_idx)]
            task = hf_item.get("task", hf_item.get("language_instruction", ""))
            
            for i in range(start_idx, end_idx):
                item = source_ds[i]
                
                # 转换图像张量为可以保存的 uint8 数组
                img_tensor = item["observation.image"]
                img_uint8 = (img_tensor.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
                
                state = item["observation.state"].numpy()
                actions = item.get("actions", item.get("action")).numpy()
                
                # 就像你之前做的那样，一帧一帧加进去
                ds_mix.add_frame({
                    "observation.image": img_uint8,
                    "observation.state": state,
                    "actions": actions,
                    "task": task,
                })
                
            ds_mix.save_episode()

    print("\n🚀 开始逐帧拷贝数据...")
    copy_episodes(ds_gs, gs_repo)
    copy_episodes(ds_real, real_repo)
    
    print(f"\n✅ 合并大功告成！新数据集保存在: {out_path}")

if __name__ == "__main__":
    main()