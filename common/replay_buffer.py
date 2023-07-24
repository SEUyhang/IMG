import numpy as np
import threading

class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        # 动作的数量
        self.n_actions = self.args.n_actions
        # 智能体的数量
        self.n_agents = self.args.n_agents
        # 状态的形状
        self.state_shape = self.args.state_shape
        # 动作的形状
        self.act_shape = self.args.act_shape
        # 观测的形状
        self.obs_shape = self.args.obs_shape
        # buffer的容量 存episode的数量
        self.size = self.args.buffer_size
        # episode的最长timesteps数
        self.episode_limit = self.args.episode_limit
        # 当前的buffer的下标(可插入的位置的坐标) 以及当前的buffer的大小
        self.current_idx = 0
        self.current_size = 0
        # 创造buffer来存储info
        self.buffers = {'o': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
                        'a': np.empty([self.size, self.episode_limit, self.n_agents, self.act_shape]),
                        's': np.empty([self.size, self.episode_limit, self.state_shape]),
                        'r': np.empty([self.size, self.episode_limit, 1]),
                        'o_next': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
                        's_next': np.empty([self.size, self.episode_limit, self.state_shape]),
                        'avail_a': np.empty([self.size, self.episode_limit, self.n_agents, self.act_shape, self.n_actions]),
                        'avail_a_next': np.empty([self.size, self.episode_limit, self.n_agents, self.act_shape, self.n_actions]),
                        'a_onehot': np.empty([self.size, self.episode_limit, self.n_agents, self.act_shape, self.n_actions]),
                        'padded': np.empty([self.size, self.episode_limit, 1]), # ？
                        'done': np.empty([self.size, self.episode_limit, 1])
                        }
        if self.args.alg == 'maven':
            self.buffers['z'] = np.empty([self.size, self.args.noise_dim])
        # 线程锁
        self.lock = threading.Lock()

    # 存episode
    def store_episode(self, episode_batch):
        # episode数
        batch_size = episode_batch['o'].shape[0]
        with self.lock:
            # 获取存储的下标
            idxs = self._get_storage_idx(inc = batch_size)
            # 存储信息
            self.buffers['o'][idxs] = episode_batch['o']
            self.buffers['a'][idxs] = episode_batch['a']
            self.buffers['s'][idxs] = episode_batch['s']
            self.buffers['r'][idxs] = episode_batch['r']
            self.buffers['o_next'][idxs] = episode_batch['o_next']
            self.buffers['s_next'][idxs] = episode_batch['s_next']
            self.buffers['avail_a'][idxs] = episode_batch['avail_a']
            self.buffers['avail_a_next'][idxs] = episode_batch['avail_a_next']
            self.buffers['a_onehot'][idxs] = episode_batch['a_onehot']
            self.buffers['padded'][idxs] = episode_batch['padded']
            self.buffers['done'][idxs] = episode_batch['done']
            if self.args.alg == 'maven':
                self.buffers['z'][idxs] = episode_batch['z']

    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffers.keys():
            temp_buffer[key] = self.buffers[key][idx]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        # 如果加入后不会溢出
        if self.current_idx + inc <= self.size:
            idx = np.arange(self.current_idx, self.current_idx+inc)
            self.current_idx += inc
        # 加入前没有溢出， 加入后溢出了
        elif self.current_idx < self.size:
            # 溢出量
            overflow = inc - (self.size - self.current_idx)
            # 没有溢出的部分的下标
            idx_a = np.arange(self.current_idx, self.size)
            # 溢出的部分直接再添加到开端
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            # 满了以后重新从队列的头部开始
            self.current_idx = overflow
        else:
            idx = np.arange(0, inc)
            self.current_idx = inc
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx