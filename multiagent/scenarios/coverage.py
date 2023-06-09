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
        num_pois = 30
        world.collaborative = True
        # reward中无人机能量损耗的占比
        world.energy_factor = 0.0
        # 添加智能体
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = False
            agent.size = 0.03
        # 添加 POI
        world.pois = [POI() for i in range(num_pois)]
        for i, poi in enumerate(world.pois):
            poi.name = 'poi %d' % i
            poi.collide = False
            poi.movable = False
            poi.size = 0.02
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
            agent.collision = 0
            agent.SNR = 0
        # 设置poi初始属性
        for i, poi in enumerate(world.pois):
            poi.color = np.array([0.25, 0.25, 0.25])
            poi.aoi = 1
        # 设置智能体初始状态
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_pos[2] = agent.height
            agent.state_p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        # 设置poi初始状态
        for i, poi in enumerate(world.pois):
            poi.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            poi.state.p_pos[2] = 0.0
            poi.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        rew = 0.0
        if self.test_collision(agent, world) != 0:
            # 发生碰撞给一个penalty
            rew -= world.collision_penalty
            # 如果发生碰撞那就只有能量损耗没有aoi的更新
            consume_energy = self.consume_energy(world.time_slot, 0, 0, world) * 1e-3
            agent.collision = 1
            agent.energy_consumption = consume_energy
            agent.SNR = 0
        else:
            distance = agent.action.u[0] * 1e3
            fly_time = distance / agent.max_speed
            # print('flytime:',fly_time)
            hover_time = world.time_slot - fly_time
            update_poi_list, K_t, poi_SNR, update_poi_uav_tmp = self.update_poi_status(agent, world.update_poi_list, hover_time, world)
            consume_energy = self.consume_energy(fly_time, hover_time, K_t, world) * 1e-3
            agent.energy_consumption = consume_energy
            agent.SNR = poi_SNR
            agent.collision = 0
        return rew

    # 每个agent都一样
    def info(self, world):
        # 该agent的新坐标加入到路径之中
        self.info = {
            'collision': [],
            'SNR': [],
            'AOI': [],
            'energy_consumption': []
        }
        for agent in world.agents:
            self.info['collision'].append(agent.collision)
            self.info['SNR'].append(agent.SNR)
            self.info['energy_consumption'].append(agent.energy_consumption)
        for poi in world.pois:
            self.info['AOI'].append(poi.aoi)
        return self.info


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


    # 返回基准数据 记录碰撞、通信、奖励等信息
    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        comm_overhead = 0
        sum_aoi = 0.0
        # 判断是否发生碰撞
        if agent.collide:
            for a in world.agents:
                if self.test_collision(a, agent):
                    rew -= 1
                    collisions += 1
        # 判断是否进行了通信
        # 返回平均aoi
        for poi in world.pois:
            sum_aoi += poi.aoi
        return (rew, collisions, comm_overhead, sum_aoi / len(world.pois))

    # 判断两个agent是否发生碰撞
    def test_collision(self, agent, world):
        # 将运动过程分成100份，判断每一小步中是否会发生碰撞
        return world.test_collision(agent)

    # 判断是否通信
    def is_comm(self, agent):
        pass

    # 判断poi的数据是否传输成功,参数是agent和poi距离、
    # 当前time_slot agent选择收集的Poi数据数目K_t、悬停(用于收集数据)的时间，当前world对象
    def judge_upload_succsess(self, square_distance, K_t, hover_time, world):
        # 设置
        # 可用的通信带宽, MHZ
        B = 0.1
        # 参考距离（1米）下的信道功率增益 dB
        beta_0 = -60
        # 噪音功率 dBm
        sigma2 = -104
        # 路径损耗指数
        alpha = 2
        # F因子莱斯衰落
        recian_factor = 0.94
        height = world.agents[0].height
        # 每个SN的平均发射功率, w, 23dbm
        Pu = 0.01
        M = world.agents[0].num_antennas
        data_size = world.data_size
        time_slot = world.time_slot

        channel_power_gain = 10 ** (beta_0 / 10) * (square_distance ** 2 + height ** 2) ** (-alpha / 2)
        available_k = M - K_t if K_t >= 2 else M  # ZF/MRC
        SNR = available_k * Pu * channel_power_gain / (10 ** ((sigma2 - 30) / 10))
        achievable_rate = B * np.log2(1 + SNR)

        required_time = data_size / achievable_rate
        # 如果悬停时间足够收集数据择返回True和信噪比
        # print('required_time', required_time)
        if required_time <= hover_time:
            return True, SNR
        else:
            return False, SNR



    # 计算无人机能量损耗
    def consume_energy(self, fly_time, hover_time, K_t, world):
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
        Power_flying = P0 * (1 + 3 * Vt ** 2 / U_tips ** 2) + \
                       P1 * np.sqrt((np.sqrt(1 + Vt ** 4 / (4 * v0 ** 4)) - Vt ** 2 / (2 * v0 ** 2))) + \
                       0.5 * d0 * rho * s0 * A * Vt ** 3
        Power_hovering = P0 + P1 + Pu * K_t

        return fly_time * Power_flying + hover_time * Power_hovering

    def update_poi_status(self, agent, update_poi_list, hover_time, world):
        # 计算UAV和poi的距离
        dist_list = []
        uav_position = agent.state.p_pos

        for i, poi in enumerate(world.pois):
            delta_pos = uav_position - poi.state.p_pos
            dist_list.append(np.sqrt(np.sum(np.square(delta_pos))))
        # 由近及远排序 并返回一个下标数组，代表原数组从小到大排序的下标
        sorted_index_list = np.argsort(dist_list)
        M = agent.num_antennas
        # 当调度1-(M-1)个poi时，成功更新的poi的下标列表
        success_k_list = [[] for _ in range(M+1)]
        # 当调度1-(M-1)个poi时，能够成功更新的aoi的数目
        success_k_len_list = [0] * M
        # 当调度1-(M-1)个poi时，平均信噪比
        success_k_SNR_list = [0] * M
        # K_t 最大是M-1
        for tmp_k in range(M):
            # 调度0个poi直接略过
            if tmp_k == 0:
                continue
            else:
                # 调度tmp_k个POI时的信噪比
                tmp_SNR_list = []
                # 计算最近的tmp_k个的信噪比
                for index in sorted_index_list[:tmp_k]:
                    distance = dist_list[index]
                    reset_ok, current_SNR = self.judge_upload_succsess(distance, tmp_k, hover_time, world)
                    if reset_ok is True:
                        success_k_list[tmp_k].append(index)
                        tmp_SNR_list.append(current_SNR)

            success_k_len_list[tmp_k] = len(success_k_list[tmp_k])
            if len(tmp_SNR_list) != 0:
                success_k_SNR_list[tmp_k] = np.mean(tmp_SNR_list)
        # 能让更新成功数最大的 调度值(agent收集poi的数量)
        K_t = np.argmax(success_k_len_list)
        for index in success_k_list[K_t]:
            world.update_poi_list.append(index)
        world.update_poi_list.append(0)
        # 成功更新的poi的序号列表， 调度数，平均信噪比， 成功更新的poi的序号列表
        return update_poi_list, K_t, success_k_SNR_list[K_t], success_k_list[K_t]

