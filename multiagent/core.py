import numpy as np


# 实体状态基类 (POI、UAV状态都属于实体状态基类)
class EntityState(object):
    def __init__(self):
        # 物理位置
        self.p_pos = None



# 智能体状态 (包括通信和内部状态)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # 通信表达
        self.c = None
        # 剩余电量
        self.energy = None

# 智能体的动作
class Action(object):
    def __init__(self):
        # 物理动作
        self.u = None
        # 通信动作
        self.c = None


class Entity(object):
    def __init__(self):
        # 名字
        self.name = ''
        # 实体的大小（半径）
        self.size = 0.05
        # 实体是否可移动
        self.movable = False
        # 实体是否可碰撞
        self.collide = True
        # 实体材质的密度
        self.density = 25.0
        # 实体的颜色
        self.color = None
        # 实体的最大速度
        self.max_speed = None

        # 实体的状态
        self.state = EntityState()
        # 初始质量？
        self.initial_mass = 1.0

    # 获取质量 可以当作属性访问
    @property
    def mass(self):
        return self.initial_mass


#  POI属性
class POI(Entity):
    def __init__(self):
        super(POI, self).__init__()
        self.aoi = 1


# agent属性
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # 智能体是否可移动
        self.movable = True
        # 智能体不能向外界发送通信消息
        self.silent = False
        # 智能体不能观察到world
        self.blind = False
        # 物理动作噪音
        self.u_noise = None
        # 通信噪音
        self.c_noise = None
        # 大小
        self.size = 0.1
        # 如果是连续动作，动作的范围(-u_range,+u_range),角度范围(-np.pi, np.pi)
        self.u_range = 0.25
        self.u_angle = np.pi
        # 状态
        self.state = AgentState()
        # 动作
        self.action = Action()
        # 要执行的脚本动作
        self.action_callback = None
        # 无人机最大速度 km/s
        self.max_speed = 25.0
        # 记录路径
        self.trace = []
        # 数据收集
        self.data_collection = None
        # 能量消耗(移动和收集数据的损耗量)
        self.energy_consumption = 0.0
        # 通信损耗
        self.comm_consumption = 0.0
        # 飞行高度 km
        self.height = 130
        # 单个无人机的天线数量
        self.num_antennas = 6
        # 当前episode该agent撞击的次数
        self.collision = 0
        # 平均信噪比
        self.SNR = 0


# multi-agent world
class World(object):
    def __init__(self):
        # 实体列表(智能体实体和POI实体列表)
        self.agents = []
        self.pois = []
        # position range
        self.range_p = 1
        # 碰撞惩罚
        self.collision_penalty = 10.0
        # 通信维度
        self.dim_c = 0
        # 位置维度
        self.dim_p = 3
        # 色彩维度
        self.dim_color = 3
        # 模拟timestep /s
        self.time_slot = 20
        # 边界长度[-bound, +bound]
        self.bound = 400
        # 物理阻尼？
        self.damping = 0.0
        # 交互响应参数？
        self.contact_force = 1e+2
        # 接触边缘
        self.contact_margin = 1e-3
        # 观测信息
        self.num_agents_obs = 0
        self.num_pois_obs = 0
        # 数据大小(POI上传成功的数据块大小)
        self.data_size = 5
        # 存储暂时的更新poi的下标
        self.update_poi_list = []


    # 返回所有world实体
    @property
    def entities(self):
        return self.agents + self.pois

    # 返回被自定义策略控制的agents
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # 返回被脚本控制的agents
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # 更新world状态,原始MPE借助力学方程构建系统，这里由于动作就是位移改变量，需要更改
    def step(self):
        # # 为脚本控制的agents设置动作
        # for agent in self.scripted_agents:
        #     agent.action = agent.action_callback(agent, self)
        # # p_force用于存储应用于实体的操作
        # p_force = [None] * len(self.entities)
        # # 作用于agents的物理操作
        # p_force = self.apply_action_force(p_force)
        # # 作用环境的操作
        # p_force = self.apply_environment_force(p_force)
        # # 整合物理状态
        # self.integrate_state(p_force)

        # 更新agents状态 p_change:position_changes
        p_change = [None] * len(self.entities)
        # 加入噪声
        p_change = self.apply_action_move(p_change)
        # 改变位置状态
        self.integrate_state(p_change)
        for agent in self.agents:
            self.update_agent_state(agent)

    # 收集对agent位移的操作距离和角度(len, angle)
    def apply_action_move(self, p_change):
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(len(agent.action.u)) * (agent.u_noise if agent.u_noise else 0.0)
                p_change[i] = agent.action.u + noise
        return p_change



    # 整合物理状态
    def integrate_state(self, p_change):
        # 记录每个agent是否发生碰撞或者出界
        flag_list = [False] * len(self.agents)
        for i, agent in enumerate(self.agents):
            dx, dy= self.cal_pos_offset(agent)
            flag_list[i] = self.test_collision(agent)
            # 如果出界就拉回来
            if flag_list[i] == 1:
                pass
            # 没出界就直接冲
            else:
                agent.state.p_pos += np.array([dx, dy, 0.0], dtype=np.float32)


    def test_collision(self, agent1):
        dx1, dy1 = self.cal_pos_offset(agent1)
        new_pos1 = agent1.state.p_pos + np.array([dx1, dy1, 0.0], dtype=np.float32)
        if (-self.bound + agent1.size < new_pos1[0] < self.bound - agent1.size) and (-self.bound + agent1.size < new_pos1[1] < self.bound - agent1.size):
            for agent2 in self.agents:
                if agent2 is agent1:
                    pass
                else:
                    dx2, dy2 = self.cal_pos_offset(agent2)
                    new_pos2 = agent2.state.p_pos + np.array([dx2, dy2, 0.0], dtype=np.float32)
                    delta_pos = new_pos1 - new_pos2
                    dist = np.sqrt(np.sum(np.square(delta_pos))).astype(np.float32)
                    dist_min = agent1.size + agent2.size
                    if dist < dist_min:
                        return 1
        # 超出边界
        else:
            return 1
        return 0

    # 根据动作计算agent的下一时刻位置
    def cal_pos_offset(self, agent):
        # 动作 u[0] = [0,1] u[1] = [-np.pi, np.pi]
        move_len = agent.action.u[0]
        move_angle = agent.action.u[1]
        dx = np.cos(move_angle) * move_len * agent.max_speed
        dy = np.sin(move_angle) * move_len * agent.max_speed
        return dx, dy

    # 更新智能体通信状态
    def update_agent_state(self, agent):
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.random(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise

