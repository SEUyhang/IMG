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
        # 实体的加速度
        self.accel = None
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
        # 如果是连续动作，动作的范围(-u_range,+u_range),角度范围(-np.pi, np.pi)
        self.u_range = 1.0
        self.u_angle = np.pi
        # 状态
        self.state = AgentState()
        # 动作
        self.action = Action()
        # 要执行的脚本动作
        self.action_callback = None
        # 无人机最大速度
        self.max_speed = 25
        # 记录路径
        self.trace = []
        # 数据收集
        self.data_collection = None
        # 能量消耗(移动和收集数据的损耗量)
        self.energy_consumption = 0.0
        # 通信损耗
        self.comm_consumption = 0.0


# multi-agent world
class World(object):
    def __init__(self):
        # 实体列表(智能体实体和POI实体列表)
        self.agents = []
        self.pois = []
        # position range
        self.range_p = 1
        # 通信维度
        self.dim_c = 0
        # 位置维度
        self.dim_p = 2
        # 色彩维度
        self.dim_color = 3
        # 模拟timestep
        self.dt = 0.1
        # 物理阻尼？
        self.damping = 0.0
        # 交互响应参数？
        self.contact_force = 1e+2
        # 接触边缘
        self.contact_margin = 1e-3
        # 观测信息
        self.num_agents_obs = 0
        self.num_pois_obs = 0

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
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_change[i] = agent.action.u + noise
        return p_change

    # 收集对环境实体的力
    def apply_environment_force(self, p_force):
        # 简单的碰撞反应
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if (b <= a):
                    continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if (f_a is not None):
                    if (p_force[a] is None):
                        p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                if (f_b is not None):
                    if (p_force[b] is None):
                        p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
        return p_force

    # 整合物理状态
    def integrate_state(self, p_change):
        # for i, entity in enumerate(self.entities):
        #     if not entity.movable:continue
        #     # 速度受到阻力影响会有折扣， damping指物理阻尼
        #     entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
        #     # 根据受力算出加速度，并改变速度的大小 v = v + a*t =v + F/m * t
        #     if(p_force[i] is not None):
        #         entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
        #     # 如果超速了，就在各个方向的分量上进行等比例缩放，限制到最大速度
        #     if entity.max_speed is not None:
        #         speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
        #         if speed > entity.max_speed:
        #             entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1])) * entity.max_speed
        #     entity.state.p_pos += entity.state.p_vel * self.dt
        for i, agent in enumerate(self.agents):
            move_len = p_change[i][0]
            move_angle = p_change[i][1]
            dx = np.cos(move_angle) * move_len
            dy = np.sin(move_angle) * move_len
            agent.state.p_pos += [dx, dy]

    # 更新智能体通信状态
    def update_agent_state(self, agent):
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.random(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise

    # 获取两个实体碰撞的力
    def get_collision_force(self, entity_a, entity_b):
        # 如果其中有一个不是碰撞体， 那么不发生碰撞
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]
        # 如果a和b是同一个实体也不发生碰撞
        if (entity_a is entity_b):
            return [None, None]
        # 计算二者之间的实际距离
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # 两个碰撞实体允许的最小距离
        dist_min = entity_a.size + entity_b.size
        # 渗透率
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]