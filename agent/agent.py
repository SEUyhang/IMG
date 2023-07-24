import numpy as np
import torch
from torch.distributions import Categorical

# 能够通信的智能体
class Agents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args
        alg = args.alg
        if alg == 'qmix':
            from policy.qmix import QMIX
            self.policy = QMIX(args)
        else:
            raise Exception("No such algorithm")
        print('Init CommAgents')

    # 针对单个agent选择动作，在rollout中对每一个agent循环进行动作选择
    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon):
        inputs = obs.copy()
        # 该智能体能选择的动作(概率不为0)的索引 因为np.nonzero返回的是元组，元组的第一个元素是索引数组
        avail_actions_index = np.nonzero(avail_actions)[0]
        # 将agent的编号改为one-hot向量
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1
        # 如果输入需要加上上一步的动作 last_action是one-hot形式
        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        # 如果网络参数共享，那么输入还要加入agent的编号加以区分 编号是one-hot形式
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))
        # agent的隐藏状态 每个agent都会维护一个属于自己的
        hidden_state = self.policy.eval_hidden[:, agent_num, :]

        # 转换输入的形状 从(input_shape,)转换到(1, input_shape) 对avail_actions作同样的操作
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)

        # 将输入和隐藏状态放到显卡上 用于后续的迭代优化
        if self.args.cuda:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()

        # 获取q值,并更新隐藏状态
        q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)

        # 根据q值获取动作
        if self.args.alg == 'coma' or self.args.alg == 'central_v' or self.args.alg == 'reinforce':
            action = self._choose_action_from_softmax(q_value.cpu(), avail_actions, epsilon)
        else:
            q_value[avail_actions == 0.0] = -float("inf") #不可采取的动作值置为负无穷，经过softmax概率就变为0
            if np.random.uniform() < epsilon:
                action = np.random.choice(avail_actions_index)
            else:
                action = np.argmax(q_value)
        return action

    def _choose_action_from_softmax(self, inputs, avail_actions, epsilon):
        # 数组 共有avail_actions.shape[-1]个元素，每个元素的值为可选动作的总和
        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])
        # 将Actor的输出通过softmax转换成概率分布
        prob = torch.nn.functional.softmax(inputs, dim=-1)
        # 加入epsilon噪音
        prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
        # 不能执行的动作概率为0
        prob[avail_actions == 0] = 0.0
        """
        不能执行的动作概率为0之后，prob中的概率和不为1，这里不需要进行正则化，因为torch.distributions.Categorical
        会将其进行正则化。要注意在训练的过程中没有用到Categorical，所以训练时取执行的动作对应的概率需要再正则化。
        """
        action = Categorical(prob).sample().long()
        return action

    # 数据收集环境没有最长的episode,全都是一样长(self.args.episode_limit)
    def _get_max_episode_len(self, batch):
        pass


    def train(self, batch, train_step, epsilon):
        max_episode_len = self.args.episode_limit
        for key in batch.keys():
            if key != 'z':
                batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step)


''' 
# 需要通信的智能体
class CommAgents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        alg = args.alg
        if alg.find('qmix') > -1:
            from policy.qmix import QMIX
            self.policy = QMIX(args)
        else:
            raise Exception("No such algorithm")
        self.args = args
        print('Init CommAgents')

    # 为一个agent,根据weights得到概率， 然后再根据epsilon选动作
    def choose_action(self, weights, avail_actions, epsilon):
        weights = weights.unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, )

'''
