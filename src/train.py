"""
This file runs the main training/val loop, etc... using Lightning Trainer    
"""
import os
from pytorch_lightning import Trainer
from argparse import ArgumentParser
import torchvision.transforms as transforms
import gym
import torch
import numpy as np

from model.plan_2_explore import Plan2Explore

def main(args):
	p2e = Plan2Explore(hparams=args)
	p2e.plan_to_explore()
	#p2e.task_adaptation()

if __name__ == '__main__':


	torch.manual_seed(0)
	np.random.seed(0)

	parser = ArgumentParser()

	# general params
	parser.add_argument("--gpu", type=int, default=0, help="number of gpus")
	parser.add_argument("--batch_size", type=int, default=64, help="size of training batch")
	parser.add_argument("--lr", type=int, default=1e-3, help="learning rate")
	parser.add_argument("--env", type=str, default="LunarLanderContinuous-v2", help="environment")
	parser.add_argument("--warm_start_size", type=int, default=100, help="number of initial random steps")
	parser.add_argument("--num_explore_steps", type=int, default=200, help="number exploration steps")
	parser.add_argument("--num_explore_episodes", type=int, default=1, help="number exploration episodes")
	parser.add_argument("--num_episodes", type=int, default=1, help="number of episodes")
	parser.add_argument("--episode_length", type=int, default=200, help="max length of an episode")
	parser.add_argument("--num_explorations", type=int, default=1, help="number of explorations")
	parser.add_argument("--accumulate_grad_batches", type=int, default=64, help="grad batches")
	

	parser.add_argument("--world_model_hid_dim", type=int, default=12, help="world model hidden dimension")
	parser.add_argument("--world_model_num_hid", type=int, default=1, help="world model hidden dimension")

	# replay buffer params
	parser.add_argument("--replay_size", type=int, default=1000, help="max size of buffer")

	# policy network params
	parser.add_argument("--in_dim", type=int, default=1e-3, help="")
	parser.add_argument("--out_dim", type=int, default=1e-3, help="")

	# value network params

	# rssm params


	# run
	args = parser.parse_args()
	main(args)


