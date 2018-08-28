import gym
import sys
import os

#janky fix to path problems
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.vpg.vpg import VPG
from util.neural_networks import mlp

def main():
    env = gym.make('CartPole-v0')
    agent = VPG(env, mlp)

    agent.train(logdir='tests/vpg')
    agent.run()

if __name__ == '__main__' : main()
