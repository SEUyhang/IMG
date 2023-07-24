import math, random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args,
                                                                                                                **kwargs)


# 对输入的观测进行编码 输入(batch_size, n_agent, obs)
class Encoder(nn.Module):
    def __init__(self, din=32, hidden_dim=128):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(din, hidden_dim)

    def forward(self, x):
        embedding = F.relu(self.fc(x))
        return embedding


# 注意力层(融合邻居信息) 输入(batch_size, n_agent, hidden_dim)
class AttModel(nn.Module):
    def __init__(self, n_node, din, hidden_dim, dout):
        super(AttModel, self).__init__()
        self.fcv = nn.Linear(din, hidden_dim)
        self.fck = nn.Linear(din, hidden_dim)
        self.fcq = nn.Linear(din, hidden_dim)
        self.fcout = nn.Linear(hidden_dim, dout)

    # mask用来屏蔽非邻居节点
    def forward(self, x, mask):
        v = F.relu(self.fcv(x))
        q = F.relu(self.fcq(x))
        k = F.relu(self.fck(x)).permute(0, 2, 1)
        # 得到一个(batch_size, n_agent, n_agent)的矩阵，表示其他agent对当前agent的重要性(非邻居节点被屏蔽)
        att = F.softmax(torch.mul(torch.bmm(q, k), mask) - 9e15 * (1 - mask), dim=2)

        out = torch.bmm(att, v)
        # out = torch.add(out,v)
        out = F.relu(self.fcout(out))
        return out


# 最后的打分环节 输入(batch_size, n_agent, hidden_Dim)
# 输出 (batch_size, n_agent, n_action)
class Q_Net(nn.Module):
    def __init__(self, hidden_dim, dout):
        super(Q_Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, dout)

    def forward(self, obs, hidden_state):
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        h = self.rnn(obs, h_in)
        q = self.fc(h)
        return q, h


class DGN(nn.Module):
    def __init__(self, args,n_agent, num_inputs, hidden_dim, num_actions):
        super(DGN, self).__init__()

        self.encoder = Encoder(num_inputs, hidden_dim)
        self.att_1 = AttModel(n_agent, hidden_dim, hidden_dim, hidden_dim)
        self.att_2 = AttModel(n_agent, hidden_dim, hidden_dim, hidden_dim)
        self.q_net = Q_Net(hidden_dim, num_actions)

    def forward(self, x, mask):
        h1 = self.encoder(x)
        h2 = self.att_1(h1, mask)
        h3 = self.att_2(h2, mask)
        q, h = self.q_net(h3)
        return q, h















