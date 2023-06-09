import torch.nn as nn
import torch
import torch.nn.functional as F
from argparse import Namespace

class QMixNet(nn.Module):
    def __init__(self, args):
        super(QMixNet, self).__init__()
        self.args = args

        # 如果生成超参数的网络有两层MLP
        if args.two_hyper_layers:
            self.hyper_w1 = nn.Sequential(nn.Linear(args.state_shape, args.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.hyper_hidden_dim, args.n_agents * args.qmix_hidden_dim)
                                          )
            self.hyper_w2 = nn.Sequential(nn.Linear(args.state_shape, args.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.hyper_hidden_dim, args.qmix_hidden_dim)
                                          )
        else:
            self.hyper_w1 = nn.Linear(args.state_shape, args.n_agents * args.qmix_hidden_dim)
            self.hyper_w2 = nn.Linear(args.state_shape, args.qmix_hidden_dim * 1)

        self.hyper_b1 = nn.Linear(args.state_shape, args.qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(args.state_shape, args.qmix_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(args.qmix_hidden_dim, 1)
                                      )
    # states的shape为(episode_num, max_episode_len， state_shape)
    # 传入的q_values是三维的，shape为(episode_num, max_episode_len， n_agents)
    def forward(self, q_values, states):
        episode_num = q_values.size(0) # 32
        q_values = q_values.view(-1, 1, self.args.n_agents) # (episode_num * max_episode_len, 1, n_agents) = (32*60, 1, 5)
        states = states.reshape(-1, self.args.state_shape) #  (episode_num * max_episode_len, state_shape)

        w1 = torch.abs(self.hyper_w1(states)) # (episode_num*max_episode_len,n_agent*qmix_hidden_dim) (32*60, 5*32)
        b1 = self.hyper_b1(states) # (episode_num*max_episode_len,qmix_hidden_dim)

        w1 = w1.view(-1, self.args.n_agents, self.args.qmix_hidden_dim)# (episode_num*max_episode_len,n_agent,qmix_hidden_dim) (32*60, 5, 32)
        b1 = b1.view(-1, 1, self.args.qmix_hidden_dim)# (episode_num*max_episode_len,1,qmix_hidden_dim) (32*60,1,32)

        hidden = F.elu(torch.bmm(q_values, w1) + b1) # (1920,1,32)

        w2 = torch.abs(self.hyper_w2(states)) # (1920.32)
        b2 = self.hyper_b2(states) # (1920,1)

        w2 = w2.view(-1, self.args.qmix_hidden_dim, 1) # (1920, 32, 1)
        b2 = b2.view(-1, 1, 1) #(1920, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2 # (1920, 1, 1)
        q_total = q_total.view(episode_num, -1, 1) # (32, 60, 1)
        return q_total

if __name__ == '__main__':
    q_values = torch.ones([32, 60, 5])
    states = torch.ones([32, 60, 16])
    args = Namespace(state_shape=16,
                     hyper_hidden_dim=32,
                     qmix_hidden_dim=32,
                     n_agents=5,
                     two_hyper_layers=True)
    net = QMixNet(args)
    out = net(q_values, states)
    print(out.size())
