import numpy as np
from multiagent.baseScenario import BaseScenario
from multiagent.core import World, Agent, POI


class Scenario(BaseScenario):

    def make_world(self):
        world = World()
        # 设置基本的world属性
        # 通信维度
        world.dim_c = 2
        # 智能体数量
        num_agents = 3
        # POI数量
        num_pois = 20
        world.collaborative = True
        # 添加智能体
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.1
        # 添加 POI
        world.pois = [POI() for i in range(num_pois)]
        for i, poi in enumerate(world.pois):
            poi.name = 'poi %d' % i
            poi.collide = False
            poi.movable = False
        # world进行初始化
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # 设置agent初始属性
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.35])
            agent.trace = []
            agent.data_collection = None
            agent.energy_consumption = 0.0
            agent.comm_consumption = 0.0
        # 设置poi初始属性
        for i, poi in enumerate(world.pois):
            poi.color = np.array([0.25, 0.25, 0.25])
            poi.aoi = 1
        # 设置智能体初始状态
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state_p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        # 设置poi初始状态
        for i, poi in enumerate(world.pois):
            poi.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            poi.state.p_vel = np.zeros(world.dim_p)

    # 返回基准数据 记录碰撞、通信、奖励等信息
    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        comm_overhead = 0
        sum_aoi = 0.0
        # 判断是否发生碰撞
        if agent.collode:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        # 判断是否进行了通信
        # 返回平均aoi
        for poi in world.pois:
            sum_aoi += poi.aoi
        return (rew, collisions, comm_overhead, sum_aoi / len(world.pois))

    # 判断两个agent是否发生碰撞
    def is_collision(self, agent1, agent2):
        if agent1 is agent2:
            return False
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # 判断是否通信
    def is_comm(self, agent):
        pass

    def reward(self, agent, world):
        return 1.0

    # 计算无人机能量损耗
    def consume_energy(self, distance, world, K_t):
        # configs
        Pu = 0.01  # 每个PoI（SN）的平均传输功率，dB（w）
        P0 = 79.8563  # 叶片剖面功率，W
        P1 = 88.6279  # 派生功率，W
        Vt = 25  # 无人机的速度，米/秒
        U_tips = 120  # 无人机转子叶片的尖端速度，m/s
        v0 = 4.03  # 悬停状态下的平均转子感应速度，m/s
        d0 = 0.6  # 机身阻力比
        rho = 1.225  # 空气的密度，kg/m^3
        s0 = 0.05  # 转子的坚固程度
        A = 0.503  # 转子盘的面积，s^2
        # 移动时间和悬停收集数据的时间
        move_time = distance / Vt
        hover_time = world.dt - move_time
        Power_flying = P0 * (1 + 3 * Vt ** 2 / U_tips ** 2) + \
                       P1 * np.sqrt((np.sqrt(1 + Vt ** 4 / (4 * v0 ** 4)) - Vt ** 2 / (2 * v0 ** 2))) + \
                       0.5 * d0 * rho * s0 * A * Vt ** 3
        Power_hovering = P0 + P1 + Pu * K_t

        return move_time * Power_flying + hover_time * Power_hovering

    # 获取以当前智能体为中心(坐标原点)所有poi的位置信息
    def observation(self, agent, world):
        entity_pos = []
        for entity in world.pois:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # poi的颜色
        entity_color = []
        for entity in world.pois:
            entity_color.append(entity.color)
        # 其他智能体的通信
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        # 当前智能体的速度、位置、poi和其他智能体的相对位置、其他智能体的通信内容作为observation
        return np.concatenate([agent.state.p_pos] + entity_pos + other_pos + comm)
