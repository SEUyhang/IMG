import sacred
from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment()
# ex.observers.append(MongoObserver())
# 带有 @ex.config 装饰器的函数中的局部变量会被 Sacred 搜集起来作为参数, 之后可以在任意函数中使用它们
@ex.config
def config():
    batch_size = 16  # int, batch size of the training
    lr = 0.001  # float, learning rate
    lr_decay = [1000, 2000]  # list, milestones for learning rate decay
    optimizer = 'adam'  # str, the optimizer of training

@ex.capture
def run(batch_size, optimizer, lr):
    print('参数 batch_size, optimizzer, lr为', batch_size, optimizer, lr)

# 带有装饰器 @ex.automain 的函数 run() 是这个脚本的主入口, 当运行该脚本时, 程序会从 train() 函数进入开始执行
@ex.automain
def main():
    run()
    run(32, 0.01)
    run(optimizer='sgd')
