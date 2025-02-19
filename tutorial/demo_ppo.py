import argparse
import gymnasium as gym
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import drl.utils as Tools
from drl.algorithm import PPO

class ActorNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(),
                                nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                nn.Linear(hidden_dim, output_dim))
    
    def forward(self, s):
        return self.net(s)
    
    def pi(self, s, softmax_dim=-1):
        x = self.forward(s)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def action(self, s, softmax_dim=-1, deterministic=False):
        prob = self.pi(s, softmax_dim)
        if deterministic:
            act = torch.argmax(prob)
            return act.item(), None
        else:
            dist = torch.distributions.Categorical(prob)
            act = dist.sample()
            log_prob = dist.log_prob(act)
            return act.item(), log_prob.item()

class CriticNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, output_dim))

    def forward(self, s):
        return self.net(s)

def eval_loop(agent, env):
    s, info = env.reset()
    done = False
    total_rew = 0
    while not done:
        a, _ = agent.action(s, deterministic=1)
        s_, r, dw, tr, info = env.step(a)
        done = (dw or tr)
        total_rew += r
        s = s_
    env.close()
    return total_rew

def eval_policy(agent, env, loop_cnt=3):
    total_rew = 0
    for _ in range(loop_cnt):
        total_rew += eval_loop(agent, env)
    return int(total_rew / loop_cnt)

def init_agent(args):
    torch.manual_seed(args.train_seed)
    torch.cuda.manual_seed(args.train_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = ActorNet(state_dim, args.hidden_dim, action_dim)
    critic = CriticNet(state_dim, args.hidden_dim, 1)
    agent = PPO(actor, critic, **vars(args.ppo_args))
    return env, agent

def eval(args):
    env = gym.make(args.env_name, render_mode="human")
    _, agent = init_agent(args)
    agent.load_model(args.save_dir, load_actor=True)
    # for i_eps in range(1000):
    i_eps = 0
    while True:
        rewards = eval_loop(agent, env, render=True)
        i_eps += 1
        print (f'eps:{i_eps}, reward:{round(rewards, 3)}')
    env.close()

def train(args):
    env, agent = init_agent(args)
    env_render = gym.make(args.env_name, render_mode = "human" if args.render else None)

    save_dir = osp.join(args.save_dir, f'ppo_{args.env_name}')
    if osp.exists(save_dir):
        import shutil
        shutil.rmtree(save_dir)        
    os.makedirs(save_dir, exist_ok=True)
    save_file = save_dir.split('/')[-1]

    env_seed = args.env_seed
    live_time = []
    total_steps = 0
    while total_steps < args.max_timesteps:
        state, info = env.reset(seed=env_seed)
        env_seed += 1

        max_len = agent.buffer.capacity()
        rews = 0
        for _ in range(max_len):
            act, log_prob = agent.action(state)
            next_state, rew, dw, tr, info = env.step(act)
            done = (dw or tr)
            rew = -30 if rew <= -100 else rew # trick for lander
            total_steps += 1

            if total_steps % args.eval_interval == 0:
                eval_rews = eval_policy(agent, env, loop_cnt=3)
                print(f'env: {args.env_name}, step: {int(total_steps / 1000)}k, rewards:{round(eval_rews, 3)}')
                live_time.append(eval_rews)
                Tools.plot(live_time, save_dir, 
                           title=f'PPO_{args.env_name}', 
                           x_label=f"Steps({int(args.eval_interval / 1000)}k)", 
                           y_label="Eval rewards", 
                           step_interval=10)
            if total_steps > 0 and total_steps % args.save_interval == 0:
                eval_policy(agent, env_render, loop_cnt=3)
                agent.save_model(save_dir, save_file, total_steps, save_actor=1, save_critic=1)

            mask = 0 if done else 1
            agent.process(s=state, a=act, r=rew, s_=next_state, l=log_prob, m=mask)
            rews += rew

            if done:
                break
            state = next_state
        pg_loss, v_loss = agent.learn()

def parse_args():
    parser = argparse.ArgumentParser(description="PPO Training Parameters")

    # 环境参数
    env_group = parser.add_argument_group("Environment Parameters")
    env_group.add_argument('--env_name', type=str, default='LunarLander-v3', help='Name of the environment')
    env_group.add_argument('--env_seed', type=int, default=42, help='Random seed for reproducibility')
    env_group.add_argument('--render', action='store_true', help='Render the environment during evaluation')

    # 通用训练参数
    train_group = parser.add_argument_group("Training Parameters")
    train_group.add_argument('--train', action='store_true', help='Train')
    train_group.add_argument('--eval', action='store_true', help='Eval')
    train_group.add_argument('--train_seed', type=int, default=42, help='Train seed')
    train_group.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension for the actor and critic networks')
    train_group.add_argument('--max_timesteps', type=int, default=1e8, help='Maximum timesteps for training')
    train_group.add_argument('--eval_interval', type=int, default=5e3, help='Evaluate agent every N timesteps')
    train_group.add_argument('--save_interval', type=int, default=5e4, help='Save model every N timesteps')
    train_group.add_argument('--save_dir', type=str, default='save', help='Directory to save models')

    # PPO 特有参数
    ppo_group = parser.add_argument_group("PPO Parameters")
    ppo_group.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer')
    ppo_group.add_argument('--clip_ratio', type=float, default=0.2, help='PPO clip ratio')
    ppo_group.add_argument('--gamma', type=float, default=0.99, help='Discount factor for rewards')
    ppo_group.add_argument('--lamda', type=float, default=0.95, help='Lambda for GAE (Generalized Advantage Estimation)')
    ppo_group.add_argument('--entropy_coef', type=float, default=0, help='Entropy regularization coefficient')
    ppo_group.add_argument('--value_coef', type=float, default=0.5, help='Value function loss coefficient')
    ppo_group.add_argument('--epochs', type=int, default=10, help='Number of PPO epochs per update')
    ppo_group.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    ppo_group.add_argument('--buffer_size', type=int, default=2048, help='Size of the replay buffer')

    args = parser.parse_args()
    ppo_args = argparse.Namespace()
    ppo_params = ['lr', 'clip_ratio', 'gamma', 'lamda', 
                  'entropy_coef', 'value_coef', 
                  'epochs', 'batch_size', 'buffer_size']
    for param in ppo_params:
        setattr(ppo_args, param, getattr(args, param))
    setattr(args, 'ppo_args', ppo_args)

    return args

# 解析参数

def main():
    args = parse_args()
    if args.train:
        train(args)
    elif args.eval:
        eval(args)

if __name__ == '__main__':
    main()
