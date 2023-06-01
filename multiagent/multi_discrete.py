import numpy as np
import gym


class MultiDiscrete(gym.Space):
    """
    - 多离散行动空间由一系列具有不同参数的离散行动空间组成
    - 它既可以适用于离散动作空间，也可以适用于连续（Box）动作空间
    - 它对于表示游戏控制器或键盘是很有用的，每个键都可以表示为一个离散的动作空间
    - 它的参数是通过为每个离散的动作空间传递一个包含[min, max]的数组来实现的
       其中，离散动作空间可以采用从`min`到`max`的任何整数（包括这两个）。
    注意：一个0的值总是需要代表NOOP动作。
    例如：任天堂游戏控制器
    - 可以被概念化为3个离散的动作空间：
        1) 方向键：Discrete5个 - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4] - 参数：Min: 0, max： 4
        2) 按钮A：Discrete 2 - NOOP[0], Pressed[1] - 参数：最小：0，最大：1： 1
        3) 按钮B: Discrete 2 - NOOP[0], Pressed[1] - 参数: min: 0, max： 1
    - 可以被初始化为
        MultiDiscrete([ [0,4], [0,1], [0,1]] )
    """

    def __init__(self, array_of_param_array):
        self.low = np.array([x[0] for x in array_of_param_array])
        self.high = np.array([x[1] for x in array_of_param_array])
        self.num_discrete_space = self.low.shape[0]

    # 在取值范围内为每个动作取样
    def sample(self):
        """ 返回一个包含每个离散行动空间的样本的数组 """
        random_array = np.random.random(self.num_discrete_space)
        return [int(x) for x in np.floor(np.multiply((self.high - self.low + 1,), random_array) + self.low)]

    # 判断动作x是否符合动作空间
    def contains(self, x):
        # .all() 方法可以用于检查可迭代对象中的所有元素是否满足特定条件
        return len(x) == self.num_discrete_space and (np.array(x) >= self.low).all() and (
                    np.array(x) <= self.high).all()

    @property
    def shape(self):
        return self.num_discrete_space

    # 在打印对象时控制的输出信息
    def __repr__(self):
        return "MultiDiscrete" + str(self.num_discrete_space)

    # 当使用==时调用该方法判断两个对象是否相等
    def __eq__(self, other):
        return np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high)


