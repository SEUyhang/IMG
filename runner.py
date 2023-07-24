import numpy as np
import os
from common.rollout import RolloutWorker, CommRolloutWorker
from network.DGN import DGN
from common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt

class Runner:
    def __init__(self, env, args):
        self.env = env
        if args.alg.find('communicate') > -1:
            self.agents = DGN(args)
            self.rolloutWorker = CommRolloutWorker(env, self.agents, args)
        self.args = args
        self.episode_rewards = []