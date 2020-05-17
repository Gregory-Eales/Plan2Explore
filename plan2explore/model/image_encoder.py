import torch


class ImageEncoder(torch.nn.Module):


	def __init__(self, in_dim, hid_dim, out_dim):

		super(ImageEncoder, self).__intit__()
		
		self.define_network()

		self.leaky_relu = torch.nn.LeakyReLU()

	def define_network(self):
		pass



	def forward(self, x):

		out = torch.Tensor(x)
		
		out = self.conv1(out)
		out = self.leaky_relu(out)
		out = self.conv2(out)
		out = self.leaky_relu(out)
		out = self.conv3(out)
		out = self.leaky_relu(out)
		out = self.conv4(out)
		out = self.leaky_relu(out)

		out = torch.flatten(out)

		return out

	def loss(self):
		pass