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
from collections import namedtuple


def init_agent(args):
    torch.manual_seed(args.train_seed)
    torch.cuda.manual_seed(args.train_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = namedtuple('model', ['pi_net', 'v_net'])
    actor = ActorNet(state_dim, args.hidden_dim, action_dim)
    critic = CriticNet(state_dim, args.hidden_dim, 1)
    agent = PPO(model(actor, critic), **args.algo)
    return env, agent

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

def eval(args):
    _, agent = init_agent(args)
    env = gym.make(args.env_name, render_mode="human")
    agent.load_model(args.save_dir, load_actor=True)
    i_eps = 0
    while True:
        rewards = eval_loop(agent, env)
        i_eps += 1
        print (f'eps:{i_eps}, reward:{round(rewards, 3)}')

def train(args):
    env, agent = init_agent(args)
    env_render = gym.make(args.env_name, render_mode = "human" if args.render else None)
    save_dir = osp.join(args.save_dir, f"{args.algo['name']}_{args.env_name}")
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

        max_len = agent.data_capacity()
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
                plot(live_time, save_dir, 
                           title=f'PPO_{args.env_name}', 
                           x_label=f"Steps({int(args.eval_interval / 1000)}k)", 
                           y_label="Eval rewards", 
                           step_interval=10)
            if total_steps % args.save_interval == 0:
                # eval_policy(agent, env_render, loop_cnt=3)
                agent.save_model(save_dir, save_file, total_steps, save_actor=1, save_critic=1)

            mask = 0 if done else 1
            # mask = 0 if dw else 1
            agent.process(s=state, a=act, r=rew, s_=next_state, l=log_prob, m=mask)
            # rews += rew
            if done:
                break
            state = next_state
        # pg_loss, v_loss = agent.learn()
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
