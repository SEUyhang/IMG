import torch
import os
from network.comix_net import CEMRecurrentAgent as RNN
from network.qmix_net import QMixNet

class COMIX:
    def __init__(self, args):
        self.n_actions =args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        # 默认的输入是观测 o
        input_shape = self.obs_shape
        # 根据参数决定CEMRecurrentAgent的输入维度
        # 如果需要上一时间步的动作
        if args.last_action:
            input_shape += self.n_actions
        # 如果所有agents共用网络，需要加上agent的序号
        if args.reuse_network:
            input_shape += self.n_agents
        self.eval_rnn = RNN(input_shape, args)      # 每个agent选动作的网络
        self.target_rnn = RNN(input_shape, args)
        self.eval_qmix_net = QMixNet(args)          # 把agents的Q值加起来的网络
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
        self.target_qmix_net.load_state_dict((self.eval_qmix_net.state_dict()))

        self.eval_parameters = list(self.eval_qmix_net.parameters()) + list(self.eval_rnn.parameters())
        if args.optimizer == 'RMS':
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)

        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中, 要为每个agent都维护一个eval_hidden
        self.eval_hidden = None
        self.target_hidden = None
        print('Init alg COMIX')

    def learn(self, batch, max_episode_len, train_step, epsilon=None):
        '''
        train_step 表示是第几次学习，用来控制更新target_net网络的参数
        在learn的时候，抽取到的batch数据是四维的，四个维度分别是:
        1--第几个episode 2--episode中第几个transition 3--第几个agent的数据 4--具体obs维度
        因为在选动作时不仅需要输入当前的inputs，还要给神经网络输入hidden_state,hidden_state和
        之前的经验相关，因此就不能随机抽取经验进行学习。所以这里一次抽取多个episode，然后一次给
        神经网络传入每个episode的同一位置的transition
        '''
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        # 把batch中的数据转化成tensor
        for key in batch.keys():
            batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        s, s_next, a, r, avail_a, avail_a_next, done = batch['s'], batch['s_next'], batch['a'], \
                                                       batch['r'], batch['avail_a'], batch['avail_a_next'], \
                                                       batch['done']
        # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习 不太理解
        mask = 1 - batch['padded'].float()

        # 得到每个agent对应的Q值， 维度为(episode个数， max_episode_len， n_agents, n_actions)
        q_evals, q_targets = self.get_q_values(batch. batch, max_episode_len)
        if self.args.cuda:
            s = s.cuda()
            a = a.cuda()
            r = r.cuda()
            s_next = s_next.cuda()
            done = done.cuda()
            mask = mask.cuda()
        # 取每个agent动作对应的Q值，并且把最后不需要的一维去掉，因为最后一维只有一个值了
        q_evals = torch.gather(q_evals, dim=3, index=a).squeeze(3)
        


    def get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            # 给obs加last_action、agent_id
            inputs, inputs_next = self._get_inputs(batch, transition_idx)


    # 给obs加last_action、agent_id
    def _get_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        obs, obs_next, a_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['a_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)
        # 给obs添加上一个last_action动作、agent编号
        if self.args.last_action:
            # 如果是第一条经验， 就让前一个动作为0向量
            if transition_idx == 0:
                inputs.append(torch.zeros_like(a_onehot[:, ]))



    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
