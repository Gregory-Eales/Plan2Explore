from __future__ import absolute_import

import gym


from model.agent import Agent
from model.trainer import Trainer


env = gym.make("LunarLanderContinuous-v2")

out_dim = env.action_space.shape[0]
in_dim = env.observation_space.shape[0]

agent = Agent(hparams=None, in_dim=in_dim, out_dim=out_dim)
trainer = Trainer(env, )


trainer.fit(agent)