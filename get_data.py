import numpy as np

def explore_dict(d, indent=0):
    """递归打印字典结构"""
    for key, value in d.items():
        prefix = "  " * indent
        if isinstance(value, dict):
            print(f"{prefix}📂 Key: {key} (Dict)")
            explore_dict(value, indent + 1)
        elif isinstance(value, np.ndarray):
            print(f"{prefix}📄 Key: {key} | Array Shape: {value.shape} | Dtype: {value.dtype}")
        else:
            print(f"{prefix}📎 Key: {key} | Type: {type(value)} | Value: {value if not isinstance(value, (list, str)) or len(str(value)) < 50 else '...'}")

# 加载数据
file_path = '/workspace/RL-co-for-gs/real_new_ctrl_pp_dice_10hz/episode_0/data.npz' 
with np.load(file_path, allow_pickle=True) as data:
    # 拿到核心字典
    content = data['data'].item()
    print("=== .npz 文件深度结构探测 ===")
    explore_dict(content)