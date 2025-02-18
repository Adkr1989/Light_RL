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

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, s):
        x = torch.tanh(self.fc1(s))
        x = torch.tanh(self.fc2(x))
        return x
        # return self.net(s)
    
    def pi(self, s, softmax_dim=0):
        x = self.forward(s)
        prob = F.softmax(self.fc3(x), dim=softmax_dim)
        return prob
            

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
save_dir = 'save/a2c_' + env_name
save_dir = os.path.join(os.path.dirname(__file__), save_dir)
save_file = save_dir.split('/')[-1]
os.makedirs(save_dir, exist_ok=True)
# writer = SummaryWriter(os.path.dirname(save_dir)+'/logs/a2c_1')

PLOT = 1
# WRITER = 0


# def sample(policy, env, max_len=None, train=1, render=0, avg=0):
#     rews = 0
#     state, info = env.reset()
#     sam_cnt = 0

#     if not max_len:
#         max_len = policy.buffer.capacity() if train else int(1e6)
#     for i in range(max_len):
#         act, log_prob = policy.action(state, train)

#         next_state, rew, dw, tr, info = env.step(act)
#         done = (dw or tr)
#         #good for LunarLander
#         rew = -30 if rew <= -100 else rew
#         sam_cnt += 1

#         if train:
#             mask = 0 if done else 1
#             policy.process(s=state, a=act, r=rew, s_=next_state, l=log_prob, m=mask)
#         if render:
#             env.render() # for self define env, you must define env.render() for visual
#         rews += rew
#         if done:
#             break
#         state = next_state
#     rews = rews / (i + 1) if avg else rews
#     return rews, sam_cnt

def evaluate_policy(policy, env, turns = 3):
    total_scores = 0
    for j in range(turns):
        s, info = env.reset()
        done = False
        while not done:
            # Take deterministic actions at test time
            a, logprob_a = policy.action(s, train=0)
            s_next, r, dw, tr, info = env.step(a)
            done = (dw or tr)

            total_scores += r
            s = s_next
    return int(total_scores/turns)

def eval():
    policy.load_model(save_dir, load_actor=1)
    for i_eps in range(1000):
        rewards = policy.sample(env, train=0, render=1)
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
    # from tqdm import tqdm
    # for i_eps in tqdm(range(policy.num_episodes)):
    total_sam_cnt = 0
    show_cnt = 0
    for i_eps in range(policy.num_episodes):
    # for i_eps in range(1000000):
        # rewards = policy.sample(env)
        # rewards, sam_cnt = sample(policy, env)
        global env_seed
        state, info = env.reset(seed=env_seed)
        env_seed += 1
        # sam_cnt = 0

        # if not max_len:
        #     max_len = policy.buffer.capacity() if train else int(1e6)
        max_len = policy.buffer.capacity()
        rews = 0
        for _ in range(max_len):
            act, log_prob = policy.action(state, train=1)

            next_state, rew, dw, tr, info = env.step(act)
            done = (dw or tr)
            #good for LunarLander
            rew = -30 if rew <= -100 else rew
            # sam_cnt += 1
            total_sam_cnt += 1

            # if train:
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
        if PLOT:
            pass
            # live_time.append(rews)
            # Tools.plot(live_time, 'PPO_'+env_name, save_dir, 1000)
        # if WRITER:
        #     writer.add_scalar('reward', reward_avg, global_step=i_eps)
        #     writer.add_scalar('loss/pg_loss', pg_loss, global_step=i_eps)
        #     writer.add_scalar('loss/v_loss', v_loss, global_step=i_eps)
        # if (i_eps + 1) % 50 == 0:
        # if total_sam_cnt % 1000 == 0:
        if total_sam_cnt > show_cnt * 5000:
            show_cnt += 1
            rews = evaluate_policy(policy, eval_env, turns=3)
            live_time.append(rews)
            Tools.plot(live_time, 'PPO_'+env_name, save_dir, 100)
            print (f'EPS:{i_eps}, step: {total_sam_cnt}, rewards:{round(rews, 3)}, pg_loss:{round(pg_loss, 3)}, v_loss:{round(v_loss, 3)}')
        if i_eps % 100000 == 0 or i_eps == (policy.num_episodes - 1):
            policy.save_model(save_dir, save_file, str(i_eps + 1), save_actor=1, save_critic=1)
    # writer.close()

if __name__ == '__main__':
    train()
    # eval()
