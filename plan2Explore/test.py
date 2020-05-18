from __future__ import absolute_import

import gym


from model.agent import Agent

agent = Agent(hparams=None)
env = gym.make("LunarLandarContinuous-v3")

