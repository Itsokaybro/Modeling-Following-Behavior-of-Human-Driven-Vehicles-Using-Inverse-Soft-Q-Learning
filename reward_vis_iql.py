import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # 用于3D图
from env import PlatoonEnv
from agent.sac import SAC

# 创建保存图片的文件夹
leader_speed = 15.0
save_dir = f'reward_vis_hh/iql/spacing=10_leader_speed={leader_speed}'
os.makedirs(save_dir, exist_ok=True)

def load_trained_agent(checkpoint_path, sac_args):
    env_args = {
        'num_vehicles': 2,
        'dt': 0.1,
        'cav_index': [1],
        'select_scenario': 0
    }
    env = PlatoonEnv(**env_args)
    obs_dim = env.observation_space.shape[0]  # 3
    action_dim = env.action_space.shape[0]      # 1
    action_range = [
        float(env.action_space.low.min()),
        float(env.action_space.high.max())
    ]
    agent = SAC(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_range=action_range,
        batch_size=sac_args['batch_size'],
        args=sac_args
    )
    agent.load(checkpoint_path)
    return agent

def compute_iql_reward(agent, state, action):
    """
    计算 IQL 奖励： Q(s,a) - gamma * V(s)
    state: numpy 数组, shape (3,)
    action: numpy 数组, shape (1,)
    """
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    action_tensor = torch.FloatTensor(action).unsqueeze(0)
    with torch.no_grad():
        Q_val = agent.critic(state_tensor, action_tensor)
        V_val = agent.getV(state_tensor)
    return (Q_val - agent.args['gamma'] * V_val).item()

def plot_contour(leader_speed, agent, save_path):
    """
    绘制等高线图：固定 leader_speed=15.0，action=0，
    横轴为 follower_speed，纵轴为 spacing
    状态为 [leader_speed, follower_speed, spacing]
    """
    #leader_speed = 15.0
    follower_speed_range = np.linspace(10, 30, 100)
    spacing_range = np.linspace(10, 45, 100)
    reward_matrix = np.zeros((len(spacing_range), len(follower_speed_range)))
    
    # 注意：这里的矩阵行对应 spacing，列对应 follower_speed
    for i, spacing in enumerate(spacing_range):
        for j, f_speed in enumerate(follower_speed_range):
            state = np.array([leader_speed, f_speed, spacing], dtype=np.float32)
            action = np.array([0.0], dtype=np.float32)
            reward_val = compute_iql_reward(agent, state, action)
            reward_matrix[i, j] = reward_val

    plt.figure(figsize=(8,6))
    # 为了使标签均匀，设置 xticks 和 yticks
    
    plt.figure(figsize=(10,10))
    ax = sns.heatmap(reward_matrix, cmap='viridis')
    x_ticks = np.arange(0, len(spacing_range), 5)
    x_labels = np.round(spacing_range[::5], 1)
    
    y_ticks = np.arange(0, len(follower_speed_range), 5)
    y_labels = np.round(follower_speed_range[::5], 1)
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Spacing')
    ax.set_ylabel('Follower Speed')
    ax.set_title(f'IQL Reward Contour (action=0, leader_speed={leader_speed})')
    plt.savefig(os.path.join(save_path, f'reward_contour_{leader_speed}.png'))
    plt.close()
    
def plot_3d_surface(leader_speed, agent, save_path):
    """
    绘制3D曲面图：同样固定 leader_speed=15.0, action=0，
    横轴 follower_speed，纵轴 spacing，z 轴为 IQL 奖励
    """
    #leader_speed = 15.0
    follower_speed_range = np.linspace(10, 30, 50)
    spacing_range = np.linspace(10, 45, 50)
    F_speed, Spacing = np.meshgrid(follower_speed_range, spacing_range)
    Reward = np.zeros_like(F_speed)
    
    for i in range(F_speed.shape[0]):
        for j in range(F_speed.shape[1]):
            state = np.array([leader_speed, F_speed[i,j], Spacing[i,j]], dtype=np.float32)
            action = np.array([0.0], dtype=np.float32)
            Reward[i,j] = compute_iql_reward(agent, state, action)
    
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(F_speed, Spacing, Reward, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Spacing')
    ax.set_ylabel('Follower Speed')
    ax.set_zlabel('IQL Reward')
    ax.set_title('IQL Reward 3D Surface (action=0, leader_speed=20.0)')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(os.path.join(save_path, f'reward_3d_surface_{leader_speed}.png'))
    plt.close()

def plot_state_slice(leader_speed, agent, fixed_spacing, save_path):
    """
    绘制状态切片图：固定 spacing=fixed_spacing, leader_speed=15.0, action=0，
    绘制不同 follower_speed 下的 IQL 奖励
    """
    #leader_speed = 15.0
    follower_speed_range = np.linspace(10, 30, 100)
    rewards = []
    for f_speed in follower_speed_range:
        state = np.array([leader_speed, f_speed, fixed_spacing], dtype=np.float32)
        action = np.array([0.0], dtype=np.float32)
        rewards.append(compute_iql_reward(agent, state, action))
    plt.figure(figsize=(8,6))
    plt.plot(follower_speed_range, rewards, marker='o')
    plt.xlabel('Follower Speed')
    plt.ylabel('IQL Reward')
    plt.title(f'IQL Reward Slice (spacing={fixed_spacing}, action=0, leader_speed={leader_speed})')
    plt.savefig(os.path.join(save_path, f'reward_slice_spacing_{fixed_spacing}_{leader_speed}.png'))
    plt.close()

def plot_scatter(leader_speed, agent, num_points, save_path):
    """
    绘制散点图：随机采样状态，并用颜色编码显示 IQL 奖励
    状态: [leader_speed, follower_speed, spacing]，leader_speed 固定为 15.0
    """
    #leader_speed = 15.0
    states = []
    rewards = []
    for _ in range(num_points):
        f_speed = random.uniform(10, 30)
        spacing = random.uniform(10, 45)
        state = np.array([leader_speed, f_speed, spacing], dtype=np.float32)
        action = np.array([0.0], dtype=np.float32)
        r = compute_iql_reward(agent, state, action)
        states.append(state)
        rewards.append(r)
    states = np.array(states)
    f_speeds = states[:,1]
    spacings = states[:,2]
    
    plt.figure(figsize=(8,6))
    sc = plt.scatter(f_speeds, spacings, c=rewards, cmap='viridis', s=50)
    plt.xlabel('Spacing')
    plt.ylabel('Follower Speed')
    plt.title(f'IQL Reward Scatter Plot (action=0, leader_speed={leader_speed})')
    plt.colorbar(sc, label='IQL Reward')
    plt.savefig(os.path.join(save_path, f'reward_scatter_{leader_speed}.png'))
    plt.close()

if __name__ == "__main__":
    sac_args = {
        'gamma': 0.99,
        'critic_tau': 0.005,
        'init_temp': 0.001,
        'hidden_dim': 128,
        'hidden_depth': 2,
        'actor_lr': 1e-4,
        'critic_lr': 1e-4,
        'alpha_lr': 0,
        'batch_size': 64,
        'device': 'cpu',
        'log_std_bounds': [-10, 2],
        'method': {
            'type': 'iq',
            'alpha': 10,
            'only_expert_states': True,
            'single_q': False,
            'use_chi2': True
        },
        'train': {
            'actor_update_freq': 2,
            'target_update_freq': 2
        }
    }
    
    # 假设训练好的模型保存在这个 checkpoint 路径下
    checkpoint_path = 'results_best_hh2veh/best_loss_model'
    agent = load_trained_agent(checkpoint_path, sac_args)
    
    # 绘制等高线图
    plot_contour(leader_speed, agent, save_dir)
    
    # 绘制 3D 曲面图
    plot_3d_surface(leader_speed, agent, save_dir)
    
    # 绘制状态切片图，固定 spacing 取中间值，比如 20
    plot_state_slice(leader_speed, agent, fixed_spacing=30, save_path=save_dir)
    
    # 绘制散点图（颜色编码）
    plot_scatter(leader_speed, agent, num_points=300, save_path=save_dir)
    
    print("All reward visualization images have been saved in the 'reward_vis' folder.")
