import numpy as np
import torch
from torch.distributions import one_hot_categorical
import time

class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init RolloutWorker')

    @torch.no_grad()
    def generate_episode(self, episode_num=None, evaluate=False):
        # prepare for save replay of evaluation
        if self.args.replay_dir != '' and evaluate and episode_num ==0:
            self.env.close()
        o, u, r, s, avail_u, u_onehot, done = [],[],[],[],[],[],[]
        self.env.reset()
        done = False
        step = 0
        # 累计奖励
        episode_reward = 0
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        while not done and step < self.episode_limit:
            obs = self.env.get_obs()
            state = self.env.get_state()
            actions, avail_actions, actions_onehot = [], [], []
            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)
                action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                   avail_action, epsilon)
                # 生成onehot向量
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(np.int(action))
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot
            reward, done, info = self.env.step(actions)
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))

# RolloutWorker for communication
class CommRolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        # 控制epsilon递减的因子
        self.anneal_epsilon = args.anneal_epsilon
        # epsilon的最小值，减到一定程度就不动了
        self.min_epsilon = args.min_epsilon
        print('Init CommRolloutWorker')

    # 生成episode
    @torch.no_grad()
    def generate_episode(self, episode_num=None, evaluate=False):
        # 为保存经验做准备,还不太懂
        if self.args.replay_dir != '' and evaluate and episode_num == 0:
            self.env.close()
        # adj是邻接矩阵
        o, u, r, s, avail_u, u_onehot, adj, done = [], [], [], [], [], [], [], []
        self.env.reset()
        done = False
        step = 0
        episode_reward = 0
        # 上一时间步每个agent动作的onehot向量
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        # NotImplemented
        self.agents.policy.init_hidden(1)
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        # 产生一个episode
        while not done and step < self.episode_limit:
            # state和obs不用step,直接用环境的API  !!!给env加一个get_state()
            obs = self.env.get_obs()
            state = self.env.get_state()
            # 获取邻接矩阵  !!!给env加一个get_adj()
            adj_matrix = self.env.get_adj()
            # 都是np数组类型
            actions, avail_actions, actions_onehot = [], [], []
            # 获取每一个agent的每一个action的权重 !!!DGN没有时序信息这个agents.get_action_weights要重新写
            weights = self.agents.get_action_weights(np.array(obs), last_action, adj)
            # 为每一个agent挑选动作
            for agent_id in range(self.n_agents):
                # !!!env.get_avail_agent_actions也要添加 获取可以选择的动作
                avail_action = self.env.get_avail_agent_actions(agent_id)
                # !!!根据可选动作和权重获取动作
                action = self.agents.choose_action(weights[agent_id], avail_actions, epsilon)
                # 产生所选动作的one_hot向量
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(np.int(action))
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot
            # obs和state不再由step提供
            reward, terminated, info = self.env.step(actions)
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            episode_reward += reward
            step += 1

            # 如果epsilon衰减速度是按照小的timestep来衰减的
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        # 计算next_obs、next_state、next_adj
        obs = self.env.get_obs()
        state = self.env.get_state()
        adj_matrix = self.env.get_adj()
        o.append(obs)
        s.append(state)
        adj.append(adj_matrix)
        o_next = o[1:]
        s_next = s[1:]
        adj_next = adj[1:]
        o = o[:-1]
        s = s[:-1]
        adj = adj[:-1]
        # 获取可用的动作(包括last obs情况的可用动作)，因为target_q在训练时需要可用动作
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # 把一个episode的内容整合到一起
        episode = dict(
            o = o.copy(),
            s = s.copy(),
            u = u.copy(),
            r = r.copy(),
            adj = adj.copy(),
            # 计算target_q时需要可用的动作
            avail_u = avail_u.copy(),
            o_next = o_next.copy(),
            s_next = s_next.copy(),
            avail_u_next = avail_u_next.copy(),
            u_onehot = u_onehot.copy(),
            terminated = terminated.copy()
        )
        # 给episode加维度
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon
        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            # 这个还不太明白什么意思
            self.env.save_replay()
            self.env.close()
        return episode, episode_reward, step

