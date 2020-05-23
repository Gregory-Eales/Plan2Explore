import torch
import pytorch_lightning as pl

from .ensamble_net import EnsambleNetwork

class DE(pl.LightningModule):

	def __init__(self, hparams):
		super(DE, self).__init__()

		self.obs_dim = self.hparams.observation_size
		self.act_dim = self.hparams.action_size
		
		self.hparams = hparams

		self.in_dim = self.obs_dim + self.act_dim
		self.out_dim = self.obs_dim 
		self.hid_dim = hparams.world_model_hid_dim
		self.num_hid = hparams.world_model_num_hid
		self.num_ensambles = hparams.num_ensambles

		self.ensamble_networks = torch.nn.ModuleDict()
		
		self.define_networks()

	def define_networks(self):
		
		for i in range(self.num_ensambles):

			self.ensamble_networks["E{}".format(i)] = EnsambleNetwork(self.hparams)
			
	def forward(self, observation, aciton):

		x = torch.cat([observation, action], dim=1)

		out = torch.Tensor(x)
		
		for key in self.layer.keys():
			out = self.layer[key](out)
			out = self.leaky_relu(out)

		return out

	def training_step(self, batch, batch_idx):
		
		s, a, _, n_s, _ = batch

		x = torch.cat([s, a], dim=1)

		prediction = self.forward(x)
		loss = F.mse_loss(prediction, n_s)
		tensorboard_logs = {'train_loss': loss}

		return {'loss': loss, 'log': tensorboard_logs}	
	
	"""
	def validation_step(self, batch, batch_idx):
		# OPTIONAL
		x, y = batch
		y_hat = self.forward(x)
		return {'val_loss': F.cross_entropy(y_hat, y)}

	def validation_epoch_end(self, outputs):
		# OPTIONAL
		avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

		tensorboard_logs = {'avg_val_loss': avg_loss}
		return {'val_loss': avg_loss, 'log': tensorboard_logs}

	def test_step(self, batch, batch_idx):
		# OPTIONAL
		x, y = batch
		y_hat = self.forward(x)
		return {'test_loss': F.cross_entropy(y_hat, y)}

	def test_epoch_end(self, outputs):
		# OPTIONAL
		avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

		tensorboard_logs = {'test_val_loss': avg_loss}
		return {'test_loss': avg_loss, 'log': tensorboard_logs}
	"""

	def configure_optimizers(self):
		# REQUIRED
		# can return multiple optimizers and learning_rate schedulers
		return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

	def train_dataloader(self):
		# REQUIRED
		"""
		return DataLoader(MNIST(os.getcwd(), train=True, download=True,
		 transform=transforms.ToTensor()), batch_size=self.hparams.batch_size)
		"""
		pass

	@staticmethod
	def add_model_specific_args(parent_parser):
		"""
		Specify the hyperparams for this LightningModule
		"""
		# MODEL specific
		parser = ArgumentParser(parents=[parent_parser], add_help=False)
		parser.add_argument('--learning_rate', default=0.02, type=float)
		parser.add_argument('--batch_size', default=32, type=int)

		# training specific (for this model)
		parser.add_argument('--max_nb_epochs', default=2, type=int)

		return parser