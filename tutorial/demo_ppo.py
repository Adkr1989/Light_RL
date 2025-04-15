import argparse
import gymnasium as gym
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from drl.algorithm import PPO
from drl.net import ActorNet, CriticNet
from drl.tools import load_config, plot
from drl.env_utils import gym_env_wrapper
from collections import namedtuple
import logging

# 设置基本配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def init_agent(args):
    torch.manual_seed(args.train_seed)
    torch.cuda.manual_seed(args.train_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    env = gym_env_wrapper(args.env.name)
    model = namedtuple('model', ['pi_net', 'v_net'])
    actor = ActorNet(env.state_dim, args.hidden_dim, env.action_dim)
    critic = CriticNet(env.state_dim, args.hidden_dim, 1)
    agent = PPO(model(actor, critic), **args.algo.to_dict())
    return env, agent

def eval_loop(agent, env):
    s = env.reset()
    done = False
    rs = 0
    while not done:
        a, _ = agent.action(s, deterministic=1)
        s_, r, done, info = env.step(a)
        rs += r
        s = s_
    env.close()
    return rs 

def eval_policy(agent, env, loop_cnt=3):
    total_rew = 0
    for _ in range(loop_cnt):
        total_rew += eval_loop(agent, env)
    return int(total_rew / loop_cnt)

def eval(args):
    _, agent = init_agent(args)
    env = gym_env_wrapper(args.env.name, render_mode="human")
    agent.load_model(args.save_dir, load_actor=True)
    i_eps = 0
    while True:
        rewards = eval_loop(agent, env)
        i_eps += 1
        logging.info(f'eps:{i_eps}, reward:{round(rewards, 3)}')

def train(args):
    env, agent = init_agent(args)
    # env_render = gym_env_wrapper(args.env.name, render_mode = "human" if args.env.render else None)
    save_dir = osp.join(args.save_dir, f"{args.algo['name']}_{args.env.name}")
    if osp.exists(save_dir):
        import shutil
        shutil.rmtree(save_dir)        
    os.makedirs(save_dir, exist_ok=True)
    save_file = save_dir.split('/')[-1]

    env.set_seed(args.env.seed)
    live_time = []
    total_steps = 0
    while total_steps < args.max_timesteps:
        state = env.reset(seed=env.get_seed())
        env.set_seed(env.get_seed() + 1)

        max_len = agent.data_capacity()
        for _ in range(max_len):
            act, log_prob = agent.action(state)
            next_state, rew, done, info = env.step(act)
            rew = -30 if rew <= -100 else rew # trick for lander
            
            total_steps += 1
            if total_steps % args.eval_interval == 0:
                eval_rews = eval_policy(agent, env, loop_cnt=3)
                logging.info(f'env: {args.env.name}, step: {int(total_steps / 1000)}k, rewards:{round(eval_rews, 3)}')
                live_time.append(eval_rews)
                plot(live_time, save_dir, 
                           title=f'PPO_{args.env.name}', 
                           x_label=f"Steps({int(args.eval_interval / 1000)}k)", 
                           y_label="Eval rewards", 
                           step_interval=10)
            if total_steps % args.save_interval == 0:
                # eval_policy(agent, env_render, loop_cnt=3)
                agent.save_model(save_dir, save_file, total_steps, save_actor=1, save_critic=1)

            mask = 0 if done else 1
            # mask = 0 if dw else 1
            agent.process(s=state, a=act, r=rew, s_=next_state, l=log_prob, m=mask)
            if done:
                break
            state = next_state
        agent.learn()

def main():
    config_yaml = '../configs/ppo.yaml'
    args = load_config(config_yaml)
    if args.train:
        train(args)
    elif args.eval:
        eval(args)

if __name__ == '__main__':
    main()
