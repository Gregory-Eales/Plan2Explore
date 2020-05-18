import gym
import pytorch_lightning as pl

from .agent import Agent
from .replay_buffer import ReplayBuffer
from .world_model import WorldModel
from .latent_disagreement_ensamble import LDE


class Plan2Explore(object):

	def __init__(self, hparams):
		
		self.hparams = hparams

		self.env = gym.make(self.hparams.env)
		self.replay_buffer = ReplayBuffer(self.hparams)
		self.world_model = WorldModel(self.hparams)
		self.lde = LDE(self.hparams)

		# exploration actor critic
		self.exp_actor_critic = Agent(self.hparams)

		# task actor critic
		self.task_actor_critic = Agent(self.hparams)

		# trainers
		self.world_model_trainer = pl.Trainer(gpus=self.hparams.gpu)
        self.lde_trainer = pl.Trainer(gpus=self.hparams.gpu)
        self.exp_ac_trainer = pl.Trainer(gpus=self.hparams.gpu)
        self.task_ac_trainer = pl.Trainer(gpus=self.hparams.gpu)

	def plan_to_explore(self):

		exploring = True

		self.initial_random_explore()
		
		for _ in range(self.hparams.num_explore_steps):
			
			self.fit_world_model()

			self.fit_lde()

			self.fit_exp_policy()

			self.explore_env()

		return self.replay_buffer, self.world_model

	def initial_random_explore(self):
		pass

	def fit_world_model(self):
		pass

	def fit_lde(self):
		pass

	def fit_exp_policy(self):
		pass

	def explore_env(self):
		pass
