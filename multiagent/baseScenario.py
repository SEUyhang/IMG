import numpy as np

# 定义情景基类 子类必须实现make_world和 reset_world
class BaseScenario(object):
    # 创建world的基础元素
    def make_world(self):
        raise NotImplementedError()
    # 创建world初始设置
    def reset_world(self, world):
        raise NotImplementedError()