import pandas as pd
import numpy as np
import pickle

def load_expert_transitions(csv_file, front_speed=15.0):
    """
    从 CSV 文件中加载专家数据，并转换为 transition 列表。
    CSV 文件格式为：
      Vehicle_ID, v_rel, spacing, action
    我们将状态定义为 [front_speed, follower_speed, spacing]，
    其中 follower_speed = front_speed + v_rel.
    奖励定义为 follower_speed 差分（即 next_follower_speed - current_follower_speed）。
    """
    df = pd.read_csv(csv_file)
    
    transitions = []
    num_rows = len(df)
    # 遍历除最后一行以外的所有行，构造 transition
    for i in range(num_rows - 1):
        # 当前行数据
        row_curr = df.iloc[i]
        v_rel_curr = float(row_curr['v_rel'])
        spacing_curr = float(row_curr['spacing'])
        action_curr = float(row_curr['action'])
        
        state = np.array([front_speed, front_speed + v_rel_curr, spacing_curr], dtype=np.float32)
        
        
        row_next = df.iloc[i + 1]
        v_rel_next = float(row_next['v_rel'])
        spacing_next = float(row_next['spacing'])
        state_next = np.array([front_speed, front_speed + v_rel_next, spacing_next], dtype=np.float32)
        
        
        action = np.array([action_curr], dtype=np.float32)
        reward = (state_next[1] - state[1])
        done = False  
        transitions.append((state, state_next, action, reward, done))
    
   
    row_last = df.iloc[-1]
    v_rel_last = float(row_last['v_rel'])
    spacing_last = float(row_last['spacing'])
    action_last = float(row_last['action'])
    state_last = np.array([front_speed, front_speed + v_rel_last, spacing_last], dtype=np.float32)
    transitions.append((state_last, state_last, np.array([action_last], dtype=np.float32), 0.0, True))
    
    return transitions

if __name__ == "__main__":
    csv_filename = 'ah_data.csv'
    transitions = load_expert_transitions(csv_filename, front_speed=15.0)
    for t in transitions[:5]:
        print(t)

    with open('./experts/expert_transitions_2veh.pkl', 'wb') as f:
        pickle.dump(transitions, f)
    print("转换后的专家数据已保存到 expert_transitions.pkl")

