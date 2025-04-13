import os
import random
import numpy as np
import torch
from collections import deque
from itertools import count
import matplotlib.pyplot as plt
import tensorboardX
import scipy.stats
import torch.optim as optim
import seaborn as sns
from env import PlatoonEnv
from agent.sac import SAC
from utils.memory import Memory
import pickle

def train_irl():
    # 设置随机种子
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    TurnOnVisual = 1
    
    # 环境参数
    env_args = {
        'num_vehicles': 2,
        'dt': 0.1,
        'cav_index': [1],
        'select_scenario': 0
    }
    
    # SAC参数
    sac_args = {
        'gamma': 0.99,  # 折扣因子
        'critic_tau': 0.005,   # 目标网络软更新系数
        'init_temp': 0.001,   # 初始温度参数 0.1
        'hidden_dim': 128,
        'hidden_depth': 2,
        'actor_lr': 1e-4, #原来的学习率都是1e-4
        'critic_lr': 1e-4,
        'alpha_lr': 0, #1e-4,
        'batch_size': 64,
        'device': 'cpu', #'cuda' if torch.cuda.is_available() else 'cpu',
        'log_std_bounds': [-10, 2],
        
        # IQ-Learn特定参数
        'method': {
            'type': 'iq',
            'alpha': 10,  # χ2散度的系数
            'only_expert_states': True,
            'single_q': False,
            'use_chi2': True
        },
                                                                                                                                                                                                                                                                                                                                              
        # 训练相关参数
        'train': {
            'actor_update_freq': 2,
            'target_update_freq': 2
        }
    }
    
    # 创建环境
    env = PlatoonEnv(**env_args)
    eval_env = PlatoonEnv(**env_args)
    
    # 创建智能体
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
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
    
    # 创建经验回放内存
    REPLAY_MEMORY = int(1e6)
    INITIAL_MEMORY = int(1e4)
    memory = Memory(REPLAY_MEMORY, seed)
    
    # 创建学习率调度器
    actor_scheduler = optim.lr_scheduler.StepLR(agent.actor_optimizer, step_size=1000, gamma=0.96)
    critic_scheduler = optim.lr_scheduler.StepLR(agent.critic_optimizer, step_size=1000, gamma=0.96)
    alpha_scheduler = optim.lr_scheduler.StepLR(agent.log_alpha_optimizer, step_size=1000, gamma=1.)

    # 加载专家数据
    #expert_memory = Memory(REPLAY_MEMORY//2, seed)
    #expert_memory.load('experts/expert_transitions_2veh_98.pkl')  # 需要实现数据加载

    # 加载专家数据
    expert_memory = Memory(REPLAY_MEMORY//2, seed)
    expert_data = pickle.load(open('experts/hh_iql_2veh.pkl', 'rb'))
    expert_memory.buffer = expert_data
    # 如果 expert_data 是一个嵌套列表，则扁平化
    if isinstance(expert_data, list) and len(expert_data) > 0 and isinstance(expert_data[0], list):
        expert_data = flatten_trajectories(expert_data)
    expert_memory.buffer = expert_data

    
    # 训练参数
    num_episodes = 50 
    eval_interval = 1    # 评估间隔
    max_steps = 2000      # 每个episode的最大步数
    
    if sac_args['method']['only_expert_states'] == False:
        num_episodes = 100

    # 训练跟踪
    rewards_history = deque(maxlen=100)
    best_eval_reward = -np.inf
    total_steps = 0
    begin_learn = False
    best_corr_pearson = -np.inf

    best_loss_value = float('inf')
    best_loss_model_path = 'results_best_hh2veh/best_loss_model'
    os.makedirs(os.path.dirname(best_loss_model_path), exist_ok=True)

    if env_args['select_scenario'] == 1:
        num_episodes = env.NGSIM_episodes
        print(f'Number of episodes: {num_episodes}')
        print('vehicle_id:', env.vehicle_id)

    # 开始训练
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        env.episiode_id = episode
        
        episode_loss = 0
        for step in range(max_steps):
            # 选择动作
            if total_steps < INITIAL_MEMORY:
                action = env.action_space.sample()
            else:
                action = agent.choose_action(state, sample=True)
            
            # 执行动作
            next_state, reward, done, _ = env.step(action, vehicle_id=episode)
            episode_reward += reward
            total_steps += 1

            # 存储经验
            memory.add((state, next_state, action, reward, done))
            
            # IQ-Learn训练
            if memory.size() > INITIAL_MEMORY:
                if not begin_learn:
                    print('Begin learning!')
                    begin_learn = True
                
                policy_batch = memory.sample(sac_args['batch_size'])
                expert_batch = expert_memory.sample(sac_args['batch_size'])
                losses = agent.iq_update(policy_batch, expert_batch, None, total_steps)
                # 添加梯度裁剪，例如限制奖励网络参数的梯度范数为10
                #torch.nn.utils.clip_grad_norm_(agent.reward_net.parameters(), max_norm=10)
                # 更新学习率
                actor_scheduler.step()
                critic_scheduler.step()
                alpha_scheduler.step()

                episode_loss += losses['iq_loss']
            
            if done:
                break
                
            state = next_state
        
        # 保存模型：每个episode都保存最新的模型
        agent.save(f'results_2veh/results_2veh_train_iql/sac_platoon_iql_lastest')
        rewards_history.append(episode_reward)
        avg_reward = np.mean(rewards_history)
        
        print(f"Episode {episode}: Loss={episode_loss:.2f}")
        
        # 定期评估
        if episode % eval_interval == 0 and episode > 10:
            eval_returns, _ = evaluate(agent, eval_env, num_episodes=5)
            mean_eval_reward = np.mean(eval_returns)
            
            # 计算奖励相关性
            pearson, spearman, iql_r, true_r = compute_reward_correlation(agent, eval_env)
            #print(f"Reward Correlation - Pearson: {pearson:.3f}, Spearman: {spearman:.3f}")
            
            if TurnOnVisual and (episode == 0 or episode % (eval_interval * 1) == 0):
                plt.figure(figsize=(10, 5))
                
                plt.subplot(1, 2, 1)
                plt.scatter(true_r, iql_r, alpha=0.5)
                plt.xlabel('True Rewards')
                plt.ylabel('IQL Rewards')
                plt.title('Reward Correlation')
                
                plt.subplot(1, 2, 2)
                plt.hist(iql_r, bins=30, alpha=0.5, label='IQL')
                plt.hist(true_r, bins=30, alpha=0.5, label='True')
                plt.xlabel('Rewards')
                plt.ylabel('Count')
                plt.title('Reward Distributions')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(f'results_2veh/results_2veh_train_iql/reward_correlation_ep{episode}.png')
                plt.close()
            
            #print(f"Evaluation: Mean Reward={mean_eval_reward:.2f}")
            
            # 保存最佳模型（基于评估奖励）改代码的时候这个也要改！
            if mean_eval_reward > best_eval_reward:
                best_corr_pearson = pearson
                best_eval_reward = mean_eval_reward
                if sac_args['method']['only_expert_states'] is True:
                    agent.save('results_best_hh2veh/sac_platoon_iql_only_expert')
                else:
                    agent.save('results_best_hh2veh/sac_platoon_iql')
        
        # 新增：判断是否当前 episode 的累计 IQ 损失更低
        if abs(episode_loss) < abs(best_loss_value) and episode > 4:
            best_loss_value = episode_loss
            # 保存当前 agent 为最小loss模型
            agent.save(best_loss_model_path)
            print(f"Episode {episode}: New best loss {best_loss_value:.2f} saved.")
    
    # 训练结束后，加载最小loss的模型进行奖励函数可视化
    print("Loading best loss agent for visualization...")
    agent.load(best_loss_model_path)
    visualize_iql_reward_function(agent, device='cpu')

def flatten_trajectories(trajectories):
    flat_list = []
    for traj in trajectories:
        for transition in traj:
            flat_list.append(transition)
    return flat_list


def evaluate(agent, env, num_episodes=1):
    """评估函数"""
    returns = []
    timesteps = []
    max_steps = 1000

    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        t = 0
        done = False
        
        for i in range(max_steps):
            action = agent.choose_action(state, sample=False)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
            t += 1
            
        returns.append(episode_reward)
        timesteps.append(t)
    
    return returns, timesteps

def compute_reward_correlation(agent, env, num_episodes=3, max_steps=3000):
    """计算IQL奖励和真实奖励的相关性
    
    Args:
        agent: SAC智能体
        env: 环境实例
        num_episodes: 评估的episode数量
        max_steps: 每个episode的最大步数
        
    Returns:
        float: 皮尔逊相关系数
        float: 斯皮尔曼相关系数
        list: IQL奖励列表
        list: 真实奖励列表
    """
    iql_rewards = []
    true_rewards = []
    
    for _ in range(num_episodes):
        state = env.reset()
        
        for _ in range(max_steps):
            action = agent.choose_action(state, sample=False)
            next_state, true_reward, done, _ = env.step(action)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                action_tensor = torch.FloatTensor(action).unsqueeze(0)
                
                current_Q = agent.critic(state_tensor, action_tensor)
                next_v = agent.getV(next_state_tensor)
                iql_reward = (current_Q - agent.args['gamma'] * next_v).item()
            iql_rewards.append(iql_reward)
            true_rewards.append(true_reward)
            if done:
                break
            state = next_state
    
    iql_rewards = np.array(iql_rewards)
    true_rewards = np.array(true_rewards)
    pearson_corr = np.corrcoef(iql_rewards, true_rewards)[0,1]
    spearman_corr = scipy.stats.spearmanr(iql_rewards, true_rewards)[0]
    
    return pearson_corr, spearman_corr, iql_rewards, true_rewards

def visualize_iql_reward_function(agent, device='cpu'):
    """
    固定 leader_speed 为 15.0，并固定动作为 0，
    在 follower_speed 和 spacing 的状态空间上可视化 IQL 奖励，
    奖励计算公式： r = Q(s, a) - gamma * V(s)
    其中 s 为状态 [leader_speed, follower_speed, spacing]（leader_speed 固定为15.0）
    """
    follower_speed_range = np.linspace(10, 30, 200)
    spacing_range = np.linspace(10, 30, 200)
    reward_matrix = np.zeros((len(follower_speed_range), len(spacing_range)))
    
    for i, f_speed in enumerate(follower_speed_range):
        for j, spacing in enumerate(spacing_range):
            state = np.array([15.0, f_speed, spacing], dtype=np.float32)
            action = np.array([0.0], dtype=np.float32)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(device)
            with torch.no_grad():
                current_Q = agent.critic(state_tensor, action_tensor)
                next_v = agent.getV(state_tensor)
                reward_val = (current_Q - agent.args['gamma'] * next_v).item()
            reward_matrix[i, j] = reward_val
            
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
    plt.xlabel('Spacing')
    plt.ylabel('Follower Speed')
    plt.title('IQL Reward Function (action fixed at 0, leader_speed=15.0)')
    plt.show()

if __name__ == "__main__":
    train_irl()

