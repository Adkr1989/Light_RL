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


from drl.algorithm import BasePolicy
from drl.utils import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPO(BasePolicy):  # option: double
    def __init__(
        self,
        # model,
        a_model,
        c_model,
        buffer_size=1000,
        actor_learn_freq=1,
        learning_rate=1e-4,
        discount_factor=0.99,
        ratio_clip=0.2,
        lam_entropy=0.01,
        gae_lamda=0.95,  # td
        batch_size=64,
        verbose=False,
        act_dim=None,
        **kwargs,
    ):
        super().__init__()
        self.__dict__.update(kwargs)
        self.lr = learning_rate
        self.eps = np.finfo(np.float32).eps.item()
        self.ratio_clip = ratio_clip
        self.lam_entropy = lam_entropy
        self.adv_norm = False  # normalize advantage, defalut=False
        self.rew_norm = False  # normalize reward, default=False
        self.schedule_clip = False
        self.schedule_adam = False

        self.actor_learn_freq = actor_learn_freq
        self._gamma = discount_factor
        self._gae_lam = gae_lamda
        self._update_iteration = 10
        self._learn_critic_cnt = 0
        self._learn_actor_cnt = 0

        self._verbose = verbose
        self._batch_size = batch_size
        self._normalized = lambda x, e: (x - x.mean()) / (x.std() + e)
        self.buffer = ReplayBuffer(buffer_size, replay=False)

        self.actor_eval = a_model.to(device).train()
        self.critic_eval = c_model.to(device).train()
        self.actor_eval_optim = optim.Adam(self.actor_eval.parameters(), lr=self.lr)
        self.critic_eval_optim = optim.Adam(self.critic_eval.parameters(), lr=self.lr)

        self.criterion = nn.SmoothL1Loss()
        self.act_dim = act_dim
        self.l2_reg = 0
        self.entropy_coef = 0
        self.entropy_coef_decay = 0.99

    def action(self, state, train=1):
        state = torch.tensor(state, dtype=torch.float32, device=device)
        # if train:
        #     self.actor_eval.train()
        # else:
        #     self.actor_eval.eval()

        # act_source = self.actor_eval(state)
        # dist = F.softmax(act_source, dim=0)
        # if train:
        #     m = Categorical(dist)
        #     act = m.sample()
        #     log_prob = m.log_prob(act)
        #     return act.item(), log_prob
        # else:
        #     a = torch.argmax(dist).item()
        #     return a, None

        with torch.no_grad():
            pi = self.actor_eval.pi(state, softmax_dim=0)
            if not train:
                a = torch.argmax(pi).item()
                return a, None
            else:
                m = Categorical(pi)
                a = m.sample().item()
                pi_a = pi[a].item()
                return a, pi_a
    
    def learn(self, i_episode=0, num_episode=100):

        self.entropy_coef *= self.entropy_coef_decay
        if not self.buffer.is_full():
            # print(f'Waiting for a full buffer: {len(self.buffer)}\{self.buffer.capacity()} ', end='\r')
            return 0, 0

        loss_actor_avg, loss_critic_avg = 0, 0

        mem = self.buffer.split(self.buffer.all_memory())
        # import pdb; pdb.set_trace()
        # if self.act_dim is None:
        #     self.act_dim = mem['a'].shape[-1]

        # S = torch.tensor(mem['s'], dtype=torch.float32, device=device)
        S = torch.from_numpy(mem['s']).float().to(device)
        # A = torch.tensor(mem['a'], dtype=torch.float32, device=device).view(-1, self.act_dim)
        A = torch.tensor(mem['a'], dtype=torch.float32, device=device).view(-1, 1)
        S_ = torch.tensor(mem['s_'], dtype=torch.float32, device=device)
        R = torch.tensor(mem['r'], dtype=torch.float32).view(-1, 1)
        Log = torch.tensor(mem['l'], dtype=torch.float32, device=device).view(-1, 1)
        # import pdb; pdb.set_trace()
        if self._verbose:
            print(f'Shape S:{S.shape}, A:{A.shape}, R:{R.shape}, S_:{S_.shape}, Log:{Log.shape}')

        with torch.no_grad():
            v_evals = self.critic_eval(S)
            end_v_eval = self.critic_eval(S_[-1])

            v_evals_np = v_evals.cpu().numpy()
            end_v_eval_np = end_v_eval.cpu().numpy()

            rewards = self._normalized(R, self.eps).numpy() if self.rew_norm else R.numpy()
            adv_gae = self.GAE(rewards, v_evals_np, next_v_eval=end_v_eval_np,
                                gamma=self._gamma, lam=self._gae_lam)
            advantage = torch.from_numpy(adv_gae).to(device).unsqueeze(-1)
            advantage = self._normalized(advantage, 1e-10) if self.adv_norm else advantage
            td_target = advantage + v_evals

        optim_iter_num = int(math.ceil(S.shape[0] / self._batch_size))

        for _ in range(self._update_iteration):

            # import pdb; pdb.set_trace()
            perm = torch.randperm(S.shape[0], device=device)
            # s, a, td_target, adv, old_prob_a = \
            #     s[perm], a[perm], td_target[perm], adv[perm], Log[perm]
            perm_s, perm_a, perm_td_tar, perm_adv, perm_old_prob_a = \
                S[perm], A[perm], td_target[perm], advantage[perm], Log[perm]

            for i in range(optim_iter_num):
                idx = slice(i * self._batch_size, min((i + 1) * self._batch_size, S.shape[0]))
                
                # critic loss
                critic_loss = (self.critic_eval(perm_s[idx]) - perm_td_tar[idx]).pow(2).mean()
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

                if self._learn_critic_cnt % self.actor_learn_freq == 0:
                    # actor_core
                    # continue
                    # mu, sigma = self.actor_eval(S)
                    # dist = Normal(mu, sigma)
                    # new_log_prob = dist.log_prob(A)

                    # act_source = self.actor_eval(perm_s[idx])
                    # dist = F.softmax(act_source, dim=-1)
                    # m = Categorical(dist)
                    # new_log_prob = m.log_prob(perm_a[idx])  # 使用 log_prob 方法
                    # pg_ratio = torch.exp(new_log_prob - perm_old_prob_a[idx])  # size = [batch_size, 1]
                    # entropy_loss = m.entropy().sum(0, keepdim=True)

                    prob = self.actor_eval.pi(perm_s[idx], softmax_dim=1)
                    entropy_loss = Categorical(prob).entropy().sum(0, keepdim=True)
                    prob_a = prob.gather(1, perm_a[idx].to(torch.int64))
                    pg_ratio = torch.exp(torch.log(prob_a) - torch.log(perm_old_prob_a[idx]))  # a/b == exp(log(a)-log(b))


                    clipped_pg_ratio = torch.clamp(pg_ratio, 1.0 - self.ratio_clip, 1.0 + self.ratio_clip)
                    surrogate_loss = -1 * torch.min(pg_ratio * perm_adv[idx], clipped_pg_ratio * perm_adv[idx])

                    # policy entropy
                    # entropy_loss = -torch.mean(torch.exp(new_log_prob) * new_log_prob)
                    # entropy_loss = -torch.mean(torch.sum(dist * torch.log(dist), dim=-1))
                    # entropy_loss = m.entropy().sum(0, keepdim=True)

                    actor_loss = surrogate_loss - self.lam_entropy * entropy_loss
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
            self.ratio_clip = 0.2 * ep_ratio

        if self.schedule_adam:
            new_lr = self.lr * ep_ratio
            # set learning rate
            # ref: https://stackoverflow.com/questions/48324152/
            for g in self.actor_eval_optim.param_groups:
                g['lr'] = new_lr
            for g in self.critic_eval_optim.param_groups:
                g['lr'] = new_lr

        loss_actor_avg /= (self._update_iteration/self.actor_learn_freq)
        loss_critic_avg /= self._update_iteration
        print(f'Train over loss_actor_avg: {loss_actor_avg}, loss_critic_avg {loss_critic_avg}', end='\r')

        return loss_actor_avg, loss_critic_avg
