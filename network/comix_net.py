import torch
import torch.nn as nn
import torch.nn.functional as F

# 使用CEM方法(不是在网络中)的agent输出的是最优动作的价值
class CEMagent(nn.Module):
    def __init__(self, input_shape, args):
        super(CEMagent, self).__init__()
        self.args = args
        # 可能是加上last_action 目前还不理解
        num_inputs = input_shape + args.n_actions
        hidden_size = args.rnn_hidden_dim

        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def get_weight_decay_weights(self):
        return {}

    def init_hidden(self):
        # 创建一个与self.fc1.weight具有相同设备和数据类型的新张量，并且其形状为(1, self.args.rnn_hidden_dim)。然后，该张量被用零值填充
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim)

    def forward(self, inputs, hidden_state, actions):
        if actions is not None:
            inputs = torch.cat([inputs, actions.contiguous().view(-1, actions.shape[-1])], dim=-1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return {"Q":q, "hidden_state":x}

# 带记忆的CEMagent(带上GRU)
class CEMRecurrentAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(CEMRecurrentAgent, self).__init__()
        self.args = args
        # 可能是加上last_action 目前还不理解
        num_inputs = input_shape + args.n_actions
        hidden_size = args.rnn_hidden_dim

        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.rnn = nn.GRUCell(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def get_weight_decay_weights(self):
        return {}

    def init_hidden(self):
        # 创建一个与self.fc1.weight具有相同设备和数据类型的新张量，并且其形状为(1, self.args.rnn_hidden_dim)。然后，该张量被用零值填充
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, actions):
        if actions is not None:
            # .contiguous()方法用于确保张量在内存中是连续存储的
            inputs = torch.cat([inputs, actions.contiguous().view(-1, actions.shape[-1])], dim=-1)
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return {"Q":q, "hidden_state":h}
