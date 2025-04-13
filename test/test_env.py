# import gym
import time
import mujoco
# import gym_rocketlander
import gymnasium as gym

# env_list = ['Pendulum-v1', 'MountainCarContinuous-v0', 'Hopper-v3', ]
# mujuco_env = ['Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2', 'Walker2d-v2']
env_name = 'LunarLander-v3'
env = gym.make(env_name)

begin = time.time()
max_step = 2048
for i in range(2):
    cnt = 0
    obs = env.reset()
    for _ in range(max_step):
        cnt += 1
        act = env.action_space.sample() # your agent here (this takes random actions)
        # print(action)
        obs, rew, done, truncated, info = env.step(act)
        if done:
            print("done", i, _, rew, act)
    print(i, cnt)

env.close()
