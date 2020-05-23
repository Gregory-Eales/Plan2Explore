import gym
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .agent import Agent
from .replay_buffer import ReplayBuffer
from .world_model import WorldModel
from .disagreement_ensamble import DE


class Plan2Explore(object):

	def __init__(self, hparams):
		
		self.hparams = hparams

		self.env = gym.make(self.hparams.env)

		self.obs_dim = self.env.observation_space.shape[0]
		self.act_dim = self.env.action_space.shape[0]


		self.replay_buffer = ReplayBuffer(self.hparams, self.obs_dim, self.act_dim)
		self.world_model = WorldModel(self.hparams, self.obs_dim, self.act_dim)
		self.de = DE(self.hparams, self.obs_dim, self.act_dim)

		# exploration actor critic
		#self.exp_actor_critic = Agent(self.hparams)

		# task actor critic
		#self.task_actor_critic = Agent(self.hparams)

		# trainers
		self.world_model_trainer = pl.Trainer(gpus=self.hparams.gpu, max_epochs=self.hparams.num_epochs)
		self.de_trainer = pl.Trainer(gpus=self.hparams.gpu, max_epochs=self.hparams.num_epochs)
		self.exp_ac_trainer = pl.Trainer(gpus=self.hparams.gpu, max_epochs=self.hparams.num_epochs)
		self.task_ac_trainer = pl.Trainer(gpus=self.hparams.gpu, max_epochs=self.hparams.num_epochs)

	def plan_to_explore(self):

		self.initial_random_explore()
		
		for _ in range(self.hparams.num_explorations):
			
			self.fit_world_model()

			self.fit_de()

			#self.fit_exp_ac()

			#self.explore_env()

		return self.replay_buffer, self.world_model

	def task_adaptation(self):

		for _ in range(self.hparams.num_adaptation_steps):

			self.distil_r()

			self.fit_task_ac()

			self.execute_task_ac()

			self.add_episodes()

		return self.task_actor_critic

	def initial_random_explore(self):
		
		for e in range(self.hparams.num_explore_episodes):

			state = self.env.reset()

			for s in range(self.hparams.num_explore_steps):

				action = self.env.action_space.sample()

				next_state, reward, terminal, info = self.env.step(action)

				self.replay_buffer.store(state, action, reward, next_state, terminal)

				state = next_state

				if terminal:
					break

		pass

	def explore_env(self):
		pass

	def execute_task_ac(self):
		pass

	def add_episodes(self):
		pass

	def distil_r(self):
		pass

	def fit_world_model(self):

		dl = DataLoader(self.replay_buffer, batch_size=self.hparams.batch_size)

		self.world_model_trainer.fit(self.world_model, train_dataloader=dl)

	def fit_de(self):
		self.de_trainer.fit(self.de, train_dataloader=self.replay_buffer)

	def fit_exp_ac(self):
		self.exp_ac_trainer.fit(self.exp_actor_critic, train_dataloader=self.replay_buffer)

	def fit_task_ac(self):
		self.task_ac_trainer.fit(self.task_actor_critic, train_dataloader=self.replay_buffer)

	def explore_env(self):
		pass
