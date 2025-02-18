# import gym
import gymnasium as gym

import os
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
# import matplotlib.pyplot as plt
import torch.nn.functional as F

import drl.utils as Tools
# from drl.algorithm import A2C
from drl.algorithm import PPO

class ActorNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(),
                                # nn.Dropout(p=0.5), 
                                nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                nn.Linear(hidden_dim, output_dim), )
    
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

class CriticNet(nn.Module): # V(s)
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, output_dim), )

    def forward(self, s):
        return self.net(s)

# env
# env_name = 'CartPole-v1'
env_name = 'LunarLander-v3'
seed = 42
'''
'CartPole-v1'
'buffer_size': 1000,
'learning_rate': 1e-2,
'num_episodes': 300

'Acrobot-v1'
'buffer_size': 3000,
'learning_rate': 1e-3,
'num_episodes': 1000
'''
# env_name = 'Acrobot-v1'
env = gym.make(env_name)
eval_env = gym.make(env_name)
# env = env.unwrapped
env.reset(seed=1)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Parameters
state_space = env.observation_space.shape[0]
action_space = env.action_space.n
hidden_dim = 64

kwargs = {
    'buffer_size': 2048,
    'learning_rate': 1e-4,
    'num_episodes': 3000000
    }

actor = ActorNet(state_space, hidden_dim, action_space)
critic = CriticNet(state_space, hidden_dim, 1)
policy = PPO(actor, critic, **kwargs)

# model save setting
save_dir = 'save/ppo_' + env_name
save_dir = os.path.join(os.path.dirname(__file__), save_dir)
save_file = save_dir.split('/')[-1]
os.makedirs(save_dir, exist_ok=True)
# writer = SummaryWriter(os.path.dirname(save_dir)+'/logs/a2c_1')

PLOT = 1
# WRITER = 0

def eval_loop(policy, env, render=False):
    s, info = env.reset()
    done = False
    total_rew = 0
    while not done:
        a, _ = policy.action(s, deterministic=1)
        s_, r, dw, tr, info = env.step(a)
        if render:
            env.render()
        done = (dw or tr)
        total_rew += r
        s = s_
    return total_rew

def eval_policy(policy, env, loop_cnt=3):
    total_rew = 0
    for _ in range(loop_cnt):
        total_rew += eval_loop(policy, env)
    return int(total_rew / loop_cnt)

def eval():
    policy.load_model(save_dir, load_actor=1)
    for i_eps in range(1000):
        rewards = eval_loop(policy, env, render=True)
        print (f'EPS:{i_eps + 1}, reward:{round(rewards, 3)}')
    env.close()

render = False
env_seed = 42
def train():
    # rm dir exist
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        import shutil
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)

    live_time = []
    total_sam_cnt = 0
    for i_eps in range(policy.num_episodes):
        global env_seed
        state, info = env.reset(seed=env_seed)
        env_seed += 1

        max_len = policy.buffer.capacity()
        rews = 0
        for _ in range(max_len):
            act, log_prob = policy.action(state)

            next_state, rew, dw, tr, info = env.step(act)
            done = (dw or tr)
            rew = -30 if rew <= -100 else rew # help for LunarLander
            total_sam_cnt += 1

            if total_sam_cnt % 5000 == 0:
                eval_rews = eval_policy(policy, eval_env, loop_cnt=3)
                print (f'EPS:{i_eps}, step: {int(total_sam_cnt / 1000)}k, rewards:{round(eval_rews, 3)}')

                live_time.append(eval_rews)
                Tools.plot(live_time, save_dir, 
                            title='PPO_'+env_name, 
                            x_label="Steps", 
                            y_label="Eval rewards",
                            step_interval=10)
                
            if total_sam_cnt % 50000 == 0:
                policy.save_model(save_dir, save_file, total_sam_cnt, save_actor=1, save_critic=1)
            
            mask = 0 if done else 1
            policy.process(s=state, a=act, r=rew, s_=next_state, l=log_prob, m=mask)
            if render:
                env.render() # for self define env, you must define env.render() for visual
            rews += rew

            if done:
                break
            state = next_state
        # print(total_sam_cnt)
        #==============learn==============
        pg_loss, v_loss = policy.learn()
        # if PLOT:
        #     live_time.append(rews)
        #     Tools.plot(live_time, save_dir, 
        #                 title='PPO_'+env_name, 
        #                 x_label="Steps", 
        #                 y_label="Eval rewards",
        #                 step_interval=1000)
        
        # if WRITER:
        #     writer.add_scalar('reward', reward_avg, global_step=i_eps)
        #     writer.add_scalar('loss/pg_loss', pg_loss, global_step=i_eps)
        #     writer.add_scalar('loss/v_loss', v_loss, global_step=i_eps)
        # if (i_eps + 1) % 50 == 0:
        # if total_sam_cnt % 1000 == 0:
        
    # writer.close()

if __name__ == '__main__':
    train()
    # eval()
