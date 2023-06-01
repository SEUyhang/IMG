import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multiagent.multi_discrete import MultiDiscrete


# 多智能体环境
# 当前代码假设在运行过程中不会创建或删除agents
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    # 根据具体的scenarios用创建一个world，scenarios中的reward,observation作为参数创建env对象中
    # shared_viewer 指是否多个agengs共享同一渲染器（他们的观测是否显示在一个窗口里）
    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):
        self.world = world
        self.agents = self.world.policy_agents
        # 设置所需gym环境属性
        self.n_agents = len(world.policy_agents)
        self.n_pois = len(world.pois)
        self.num_action = 2
        # 不知道是啥
        self.n_pois_obs = world.num_agents_obs
        self.n_agents_obs = world.num_pois_obs
        # 具体scenarios对应的回调函数（reward,observation，reset等）
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # 环境参数
        self.discrete_action_space = False
        # 如果True 动作的值为 0...N,否则是一个N维的独热码
        self.discrete_action_input = False
        # 如果为true 即使动作是连续的，动作也将被离散执行
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # 如果为true 所有agent都有相同的reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.range_p = world.range_p
        self.dim_p = world.dim_p
        self.time = 0
        # 配置空间
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # 配置物理动作空间
            if self.discrete_action_space:
                # 动作空间数等于方向数乘2加1(空动作)
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,),
                                            dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)
            # 配置通信动作空间
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)
            # 对action_space 进行汇总
            if len(total_action_space) > 1:
                # 如果动作空间全是离散的，简化为MultiDiscrete动作空间
                if all([isinstance(act_space, spaces.Discrete) for act_apace in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_apace in total_action_space])
                else:
                    '''
                    Example usage:
                    self.observation_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(3)))
                    '''
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # 配置观测空间
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)

        # 每个智能体的观测是否共享同一个窗口
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            # self.n 是什么？好像是agents的数量
            self.viewers = [None] * self.n

            # bound

    def bound(self, x):
        pass

    # 与环境交互得到obs_n,reward_n, done_n,info_n
    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        # 为每一个agent设置动作
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # 更新world
        self.world.step()
        # 为每个智能体记录观测
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))
            info_n['n'].append(self._get_info(agent))
        # 所有agents在合作情况下获得相同的总体奖励
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n_agents
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        # 重置world
        self.reset_callback(self.world)
        # 重置渲染
        self._reset_render()
        # 记录每个agents的观测
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    '''
    如果是离散动作，强化学习的网络输出的是一个one-hot类型的数组，但是放在world里去执行需要
    转化为轴方向的动作，比如二维情况下，one-hot数组中的值需要转化成world.agent.action.u[0]和
    world.agent.action.u[1]的值去执行
    如果是连续动作，强化学习网络输出的是一个单一的值（如果强制输入离散值）
    '''

    # 将强化学习agent得到的动作为环境中具体的agent设置环境动作
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # 处理动作
        if isinstance(action_space, MultiDiscrete):
            # act是二维列表
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                # 加入的是one-hot 形式 ？
                act.append(action[index:(index + s)])
                index += s
            action = act
        else:
            action = [action]

        if agent.movable:
            # 物理动作
            # 如果是离散的动作值输入， 每个动作对应一个整数值
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # 处理离散动作 agent.action.u是一个dim_p维数组，每一个维度的数值控制在该方向移动距离,比如u[0]=1代表往上，u[0]=-1代表往下
                # action是一个数组，第一个值代表物理动作，第二个值代表通信动作
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            # 输入是 one-hot类型的，也就是每一个动作都是一个数组
            else:
                # 如果强制性输入离散的动作，选值最大的那个
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                # 如果是离散动作空间，分别在x,y轴加上对应的分量
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                # 如果是连续动作空间，直接把数组赋值
                else:
                    agent.action.u = action[0]
            # 对动作(力)的敏感程度，如果动作就是移动的距离就不需要了
            # sensitivity = 5.0
            # if agent.accel is not None:
            #     sensitivity = agent.accel
            # agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # 通信动作
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]
        # 确保动作全都用到了
        assert len(action) == 0

    # 获取信息用于基准测试
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # 获取某一个agent的观测
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # 获取某个agent是否完成信息
    # 现在还没有使用 -- 代理人被允许超越观察屏幕
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # 获取某一个agent的reward
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # 重置渲染
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # 渲染环境
    def render(self, mode='human'):
        if mode == 'human':
            # alphabet中的每个字符对应一个离散动作
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            # message用来记录智能体之间的通信
            message = ''
            # 对每个agent，遍历其他agents对agent的通信
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent:
                        continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + 'to' + agent.name + ':' + word + '  ')

        for i in range(len(self.viewers)):
            # 创建渲染器实例（如果需要）
            if self.viewers[i] is None:
                # import rendering只有昂我们需要的时候
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(700, 700)

        # 创建渲染几何图形
        if self.render_geoms is None:
            from multiagent import rendering
            # 图形列表和变换
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                # 实体都是画的圈
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # 往视窗里加图形
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from multiagent import rendering
            # 更新边界，使之以代理人为中心
            # agent 的观测范围 摄像机将以智能体的位置为中心，并在 x 轴和 y 轴上分别延伸 1 个单位的距离，形成一个边长为 2 的正方形范围。
            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0] - cam_range, pos[0] + cam_range, pos[1] - cam_range, pos[1] + cam_range)
            # 更新图形位置
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # 将渲染显示或加入数组
            results.append(self.viewers[i].render(return_rgb_array=(mode == 'rgb_array')))

        return results

    # 在MPE中的感受野用于定义智能体的感知范围，也称为接收器（receptor）的位置
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # 圆形感受野
        if receptor_type == 'polar':
            # 感受野的角度划分 endpoint=False参数表示不包含结束值，即不包括π
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                # 感受野的距离划分 它会生成一个包含3个元素的数组，这些元素均匀分布在起始值和结束值之间
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
                # 加入原点
                dx.append(np.array([0.0, 0.0]))
        # 网格感受野
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, + range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x, y]))
        return dx