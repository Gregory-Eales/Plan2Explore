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

		self.obs_dim = self.env.observation_space.shape[0]
		self.act_dim = self.env.action_space.shape[0]


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

		self.initial_random_explore()
		
		for _ in range(self.hparams.num_explore_steps):
			
			self.fit_world_model()

			self.fit_lde()

			self.fit_exp_ac()

			self.explore_env()

		return self.replay_buffer, self.world_model

	def task_adaptation(self):

		for _ in range(self.hparams.num_adaptation_steps):

			self.distil_r()

			self.fit_task_ac()

			self.execute_task_ac()

			self.add_episodes()

		return self.task_actor_critic

	def initial_random_explore(self):
		
		# for each episode

			# state = env.reset()

			# for each step in an episode

				# get action from rand agent

				# next_state, reward, terminal = env.act(action)

				# store trajectory

				# state = next_state

				# if terminal:

					# end episode, discount rewards

	def fit_world_model(self):
		self.world_model_trainer.fit(self.world_model, train_dataloader=self.replay_buffer)

	def fit_lde(self):
		self.lde_trainer.fit(self.lde, train_dataloader=self.replay_buffer)

	def fit_exp_ac(self):
		self.exp_ac_trainer.fit(self.exp_actor_critic, train_dataloader=self.replay_buffer)

	def fit_task_ac(self):
		self.task_ac_trainer.fit(self.task_actor_critic, train_dataloader=self.replay_buffer)

	def explore_env(self):
		pass
