import torch
import os
from network.base_net import RNN
from network.qmix_net import QMixNet

class QMIX:
    def __init__(self, args):
        # 可选的动作的数量，复合的动作需要额外改
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        input_shape = self.obs_shape
        # 根据参数决定RNN的输入维度
        # 如果上一步的动作作为输入
        if args.last_action:
            input_shape += self.n_actions
        # 如果重用网络需要加上agent编号
        if args.reuse_network:
            input_shape += self.n_agents

        # 定义网络
        self.eval_rnn = RNN(input_shape, args)  # 每个agent选动作的网络
        self.target_rnn = RNN(input_shape,args)
        self.eval_qmix_net = QMixNet(args)      # 把agents的Q值加起来的网络
        self.target_qmix_net = QMixNet(args)
        self.args = args
        if self.args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_qmix_net.cuda()
            self.target_qmix_net.cuda()

        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map
        # 如果存在模型则加载模型
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
                path_rnn = self.model_dir + '/rnn_net_params.pkl'
                path_qmix = self.model_dir + '/qmix_net_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                self.eval_qmix_net.load_state_dict(torch.load(path_qmix, map_location=map_location))
                print('Successfully load the model: {} and {}'.format(path_rnn, path_qmix))
            else:
                raise Exception("No model!")

        # 让target_net和eval_net的网络参数相同
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())
        self.eval_parameters = list(self.eval_qmix_net.parameters()) + list(self.eval_rnn.parameters())

        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)

        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden、target_hidden
        self.eval_hidden = None
        self.target_hidden = None
        print('Init alg QMIX')

    # 用经验学习参数 train_step表示是第几次学习，用来控制更新target_net网络的参数
    def learn(self, batch, max_episode_len, train_step, epsilon=None):
        '''
        在learn的时候，抽取到的数据是四维的
        1--第几个episode
        2--episode中第几个transition
        3--第几个agent的数据
        4--具体的数据(obs,act,r等)
        因为在选择动作时不仅需要输入当前的inputs，还要给神经网络输入hidden_state,hidden_state和
        之前的经验有关，因此不能随机抽取经验学习。所以这里一次抽取多个episode，然后一次传入每个episode
        的同一位置的transition
        '''
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        # 把batch里的数据转换成tensor
        for key in batch.keys():
            batch[key] = torch.tensor(batch[key], dtype=torch.float32)

        s, s_next, u, r, avail_u, avail_u_next, done = batch['s'], batch['s_next'], batch['u'], \
                                                       batch['r'], batch['avail_u'], batch['avail_u_next'],\
                                                       batch['done']
        # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习
        mask = 1 - batch["padded"].float()


    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        self.eval_hidden = torch.zeros([episode_num, self.n_agents, self.args.rnn_hidden_dim])
        self.target_hidden = torch.zeros([episode_num, self.n_agents, self.args.rnn_hidden_dim])

