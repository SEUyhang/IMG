import numpy as np
import multiagent.make_env as ma_env
import time
# 创建MPE环境
env = ma_env.make_env('coverage')

# 重置环境
obs_n = env.reset()

start = time.time()

# 执行10个时间步的动作
for _ in range(5000):
    # 生成随机动作
    act_n = [env.action_space[i].sample() for i in range(env.n_agents)]

    # 执行动作并获取下一个观察值、奖励和是否结束的信息
    obs_n, reward_n, done_n, info_n = env.step(act_n)


    # 打印当前观察值、奖励和是否结束的信息
    # print("Observations:", obs_n)
    # print("Rewards:", reward_n)
    # print("Done:", done_n)
    # print("-----------------------")
    # env.render()

# 关闭环境
env.close()

end = time.time()
print(end - start)