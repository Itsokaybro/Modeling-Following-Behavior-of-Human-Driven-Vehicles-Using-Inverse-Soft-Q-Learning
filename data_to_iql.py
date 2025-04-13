import pandas as pd
import numpy as np
import pickle

def load_expert_trajectories(csv_file):
    """
    从 CSV 文件中加载专家数据，将相同 Vehicle_ID 的行数据归为一条轨迹，
    并对每条轨迹仅取前 99 行数据，生成 98 个 transition，
    每个 transition 格式为:
        (state, next_state, action, reward, done)
    状态定义为 [lv_v, fv_v, spacing]（不做转换，直接使用 CSV 中的数值），
    奖励定义为：下一行 fv_v 与当前行 fv_v 的差值，
    除最后一条 transition 的 done 为 True，其余均为 False。
    """
    df = pd.read_csv(csv_file)
    trajectories = []
    
    # 按 Vehicle_ID 分组
    groups = df.groupby('Vehicle_ID')
    for vid, group in groups:
        # 按原始顺序排序（这里使用行索引顺序）
        group = group.reset_index(drop=True)
        # 只处理行数>=99的轨迹（生成 98 个 transition）
        if len(group) < 99:
            continue
        # 取前 99 行数据
        group = group.iloc[:99]
        traj = []
        num = len(group)  # 应该为 99
        for i in range(num - 1):  # 生成 98 个 transition
            # 当前行
            row_curr = group.iloc[i]
            lv_v_curr = float(row_curr['lv_v'])
            fv_v_curr = float(row_curr['fv_v'])
            spacing_curr = float(row_curr['spacing'])
            action_curr = float(row_curr['action'])
            # 状态：直接取 CSV 数值：[lv_v, fv_v, spacing]
            state = np.array([lv_v_curr, fv_v_curr, spacing_curr], dtype=np.float32)
            
            # 下一行
            row_next = group.iloc[i+1]
            lv_v_next = float(row_next['lv_v'])
            fv_v_next = float(row_next['fv_v'])
            spacing_next = float(row_next['spacing'])
            next_state = np.array([lv_v_next, fv_v_next, spacing_next], dtype=np.float32)
            
            # 奖励定义为 follower speed 差分，即 fv_v_next - fv_v_curr
            reward = fv_v_next - fv_v_curr
            action_arr = np.array([action_curr], dtype=np.float32)
            done = False
            # 如果是最后一个 transition，则 done 为 True
            if i == num - 2:
                done = True
            traj.append((state, next_state, action_arr, reward, done))
        trajectories.append(traj)
    
    return trajectories

if __name__ == "__main__":
    csv_filename = 'hh_data_2.csv'  
    trajectories = load_expert_trajectories(csv_filename)
    
    print("Number of trajectories:", len(trajectories))
    
    if trajectories:
        print("Sample transitions from first trajectory:")
        for t in trajectories[0][:3]:
            print(t)
    
    
    pkl_filename = './experts/hh_iql_2veh.pkl'
    with open(pkl_filename, 'wb') as f:
        pickle.dump(trajectories, f)
    print(f"Expert trajectories saved to {pkl_filename}")
