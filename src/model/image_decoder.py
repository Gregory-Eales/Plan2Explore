import torch


class StateDecoder(torch.nn.Module):


	def __init__(self):

		super(ImageDecoder, self).__intit__()

		self.conv_trans1 = torch.nn.ConvTranspose2d(in_channels, 128, 5, stride=2)
		self.conv_trans2 = torch.nn.ConvTranspose2d(in_channels, 64, 5, stride=2)
		self.conv_trans3 = torch.nn.ConvTranspose2d(in_channels, 32, 6, stride=2)
		self.mean = torch.nn.ConvTranspose2d(in_channels, 32, 6, stride=2)

		self.leaky_relu = torch.nn.LeakyReLU()

	def forward(self, x):
		
		out = torch.Tensor(x)

		out = self.conv_trans1(out)
		out = self.leaky_relu(out)
		out = self.conv_trans2(out)
		out = self.leaky_relu(out)
		out = self.conv_trans3(out)
		out = self.leaky_relu(out)
		out = self.conv_trans4(out)
		out = self.leaky_relu(out)

		return out

	def loss(self):
		pass