import torch


class ValueNetwork(torch.nn.Module):


	def __init__(self, in_dim, out_dim, hid_dim=128, num_hid=1):

		super(ValueNetwork, self).__init__()

		self.in_dim = in_dim
		self.hid_dim = hid_dim
		self.out_dim = out_dim
		self.num_hid = num_hid

		self.layer = torch.nn.ModuleDict()
		
		self.define_network()

	def define_network(self):

		self.layer["l1"] = torch.nn.Linear(self.in_dim, self.hid_dim)
		
		for i in range(2, self.num_hid+2):
			self.layer["l{}".format(i)] = torch.nn.Linear(self.hid_dim, self.hid_dim)

		self.layer["l{}".format(self.num_hid+2)] = torch.nn.Linear(self.hid_dim, self.out_dim)

		self.leaky_relu = torch.nn.LeakyReLU()

	def forward(self, x):

		out = torch.Tensor(x)
		
		for key in self.layer.keys():
			out = self.layer[key](out)
			out = self.leaky_relu(out)

		return out


	def loss(self):
		pass

def main():
	p = ValueNetwork(3, 3, 3, 5)
	p.forward(torch.ones(10, 3))


if __name__ == "__main__":
	main()