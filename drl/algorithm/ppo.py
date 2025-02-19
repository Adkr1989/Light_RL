import os
import numpy as np
from copy import deepcopy
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset


from drl.algorithm import BasePolicy
from drl.utils import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPO(BasePolicy):  # option: double
    def __init__(
        self,
        # model,
        a_model,
        c_model,
        lr=1e-4,
        epochs=10,
        batch_size=64,
        buffer_size=2048,
        clip_ratio=0.2,
        gamma=0.99,
        lamda=0.95,
        entropy_coef=0.01,
        entropy_coef_decay=0.99,
        value_coef=0.5,
        l2_reg=0,
        actor_learn_freq=1,
        verbose=False,
        act_dim=None,
        **kwargs,
    ):
        super().__init__()
        self.__dict__.update(kwargs)

        self.lr = lr
        self.eps = np.finfo(np.float32).eps.item()
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.l2_reg = l2_reg
        self.entropy_coef_decay = entropy_coef_decay

        self._gamma = gamma
        self._lamda = lamda
        self._update_epochs = epochs
        self._batch_size = batch_size

        self._verbose = verbose
        self._learn_critic_cnt = 0
        self._learn_actor_cnt = 0
        self._actor_learn_freq = actor_learn_freq
        self._normalized = lambda x, e: (x - x.mean()) / (x.std() + e)
        self.buffer = ReplayBuffer(buffer_size, replay=False)
        self.actor_eval = a_model.to(device).train()
        self.critic_eval = c_model.to(device).train()
        self.actor_eval_optim = optim.Adam(self.actor_eval.parameters(), lr=self.lr)
        self.critic_eval_optim = optim.Adam(self.critic_eval.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()

        self.act_dim = act_dim
        self.adv_norm = False  # normalize advantage, defalut=False
        self.rew_norm = False  # normalize reward, default=False
        self.schedule_clip = False
        self.schedule_adam = False

    def action(self, state, deterministic=False, **kwargs):
        state = torch.tensor(state, dtype=torch.float32, device=device)
        if deterministic:
            self.actor_eval.eval()
        else:
            self.actor_eval.train()
        with torch.no_grad():
            act = self.actor_eval.action(state, deterministic=deterministic, **kwargs)
            return act
    
    def learn(self, i_episode=0, num_episode=100):
        if not self.buffer.is_full():
            # print(f'Waiting for a full buffer: {len(self.buffer)}\{self.buffer.capacity()} ', end='\r')
            return 0, 0
        
        loss_actor_avg, loss_critic_avg = 0, 0
        self.entropy_coef *= self.entropy_coef_decay

        mem = self.buffer.split(self.buffer.all_memory())
        S = torch.from_numpy(mem['s']).float().to(device)
        A = torch.from_numpy(mem['a']).float().view(-1, 1).to(device)
        S_ = torch.from_numpy(mem['s_']).float().to(device)
        R = torch.from_numpy(mem['r']).float().view(-1, 1)
        M = torch.from_numpy(mem['m']).float().view(-1, 1)
        Log = torch.from_numpy(mem['l']).float().view(-1, 1).to(device)

        if self._verbose:
            print(f'Shape S:{S.shape}, A:{A.shape}, R:{R.shape}, S_:{S_.shape}, Log:{Log.shape}')

        with torch.no_grad():
            v_evals = self.critic_eval(S)
            end_v_eval = self.critic_eval(S_[-1])

            v_evals_np = v_evals.cpu().numpy()
            end_v_eval_np = end_v_eval.cpu().numpy()

            masks = M.numpy()
            rewards = self._normalized(R, self.eps).numpy() if self.rew_norm else R.numpy()
            adv_gae = self.GAE(rewards, v_evals_np, next_v_eval=end_v_eval_np,
                               masks=masks, gamma=self._gamma, lam=self._lamda)
            advantage = torch.from_numpy(adv_gae).to(device).view(-1, 1)
            advantage = self._normalized(advantage, 1e-10) if self.adv_norm else advantage
            v_target = advantage + v_evals

        dataset = TensorDataset(S, A, v_target, advantage, Log)
        dataloader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)

        for _ in range(self._update_epochs):

            for batch_data in dataloader:
                bs_s, bs_a, bs_v_tar, bs_adv, bs_logp_old = batch_data
                # critic loss
                critic_loss = self.criterion(self.critic_eval(bs_s), bs_v_tar).mean()
                #
                for name, param in self.critic_eval.named_parameters():
                    if 'weight' in name:
                        critic_loss += param.pow(2).sum() * self.l2_reg
                loss_critic_avg += critic_loss.item()

                self.critic_eval_optim.zero_grad()
                critic_loss.backward()
                self.critic_eval_optim.step()

                self._learn_critic_cnt += 1
                if self._verbose:
                    print(f'=======Learn_Critic_Net, cnt{self._learn_critic_cnt}=======')

                if self._learn_critic_cnt % self._actor_learn_freq == 0:
                    # actor_core
                    # continue
                    # mu, sigma = self.actor_eval(S)
                    # dist = Normal(mu, sigma)
                    # logp = dist.log_prob(A)

                    # [batch_size, act_dim]
                    prob = self.actor_eval.pi(bs_s)
                    dist = Categorical(prob)
                    pi_ent = dist.entropy().sum(0, keepdim=True)
                    logp = dist.log_prob(bs_a.flatten()).unsqueeze(-1)

                    pg_ratio = torch.exp(logp - bs_logp_old)
                    clipped_pg_ratio = torch.clamp(pg_ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                    surr1 = pg_ratio * bs_adv
                    surr2 = clipped_pg_ratio * bs_adv
                    # actor loss
                    actor_loss = -1 * torch.min(surr1, surr2) - self.entropy_coef * pi_ent    
                    #                 
                    actor_loss = actor_loss.mean()
                    loss_actor_avg += actor_loss.item()
                    
                    self.actor_eval_optim.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor_eval.parameters(), 40)
                    self.actor_eval_optim.step()
                    
                    self._learn_actor_cnt += 1
                    if self._verbose:
                        print(f'=======Learn_Actort_Net, cnt{self._learn_actor_cnt}=======')

        self.buffer.clear()
        assert self.buffer.is_empty()

        # update param
        ep_ratio = 1 - (i_episode / num_episode)
        if self.schedule_clip:
            self.clip_ratio = 0.2 * ep_ratio

        if self.schedule_adam:
            new_lr = self.lr * ep_ratio
            # set learning rate
            # ref: https://stackoverflow.com/questions/48324152/
            for g in self.actor_eval_optim.param_groups:
                g['lr'] = new_lr
            for g in self.critic_eval_optim.param_groups:
                g['lr'] = new_lr

        loss_actor_avg /= (self._update_epochs/self._actor_learn_freq)
        loss_critic_avg /= self._update_epochs
        # print(f'Train over loss_actor_avg: {loss_actor_avg}, loss_critic_avg {loss_critic_avg}')

        return loss_actor_avg, loss_critic_avg
