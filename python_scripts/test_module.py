import numpy as np
import random
'''
AlexNet

Sparse mode

Weight Sparsity:

conv1:	   87%
conv2:	 2.99%
conv3:	 1.94%
conv4:	 2.44%
conv5:	 2.98%

Activation Spasity:

conv1:	   98%
conv2:	 76.8%
conv3:	   51%
conv4:	   48%
conv5:	   47%

'''

#Class for compress data
class SRAM():
	def __init__(self, in_SRAM, size, first_in_SRAM):
		self.in_SRAM = 0
		self.size = 0
		self.first_in_SRAM = 1
		self.next = None


class Linkedlist():
	"""
	This is the linked list version for SRAM
	"""
	def __init__(self):
		self.head = None
		self.tail = None

	def add_list_item(self, item):
		item.in_SRAM = 1
		if self.head is None:
			self.head = item
		else:
			self.tail.next = item
		self.tail = item

	def remove_head(self):
		node = self.head
		self.head = node.next
		node.in_SRAM = 0
		node.next = None

	def data_num(self):
		node = self.head
		count = 0
		while node:
			count = count + node.size
			node = node.next

		return count
		




SPARTSITY_THRESHOLD = 0.5

def apply_padding(raw_IF_size, padding_size = 0):
	"""
	This function adjusts the IF_size based on padding_size number
	"""
	IF_size = raw_IF_size + 2 * padding_size
	return IF_size


def construct_IF_map(IF_size, channel_size):
	"""
	This function constructs all zeros matrix for IF map
	"""
	Input_fmap = np.zeros((channel_size, IF_size, IF_size), dtype = int)

	return Input_fmap

def construct_weight_map(weight_kernel_number, channel_size, weight_size):
	"""
	This function constructs all zeros matrix for IF map
	"""
	weight_map = np.zeros((weight_kernel_number, channel_size, weight_size, weight_size), dtype = int)

	return weight_map

def sparsify_IF_map(IF_size, raw_IF_size, channel_size, IF_sparsity, padding_size):
	"""
	This function adds randomness based on layer sparsity to the IF map
	Should avoid affecting the border padded data which should always be zero

	Calculate new sparsity by taking into consideration of the zero borders,
	so the central parts should have less non zero terms 
	Non-zero terms = IF_sparsity * IF_size ^ 2 = new_sparsity * raw_size ^ 2
	"""

	all_zeros_IF_map = construct_IF_map(IF_size, channel_size)

	new_sparsity = IF_sparsity * (IF_size) ** 2 / (raw_IF_size) ** 2

	random_number = random.randint(0,99)

	for i in range(channel_size):
		for j in range(padding_size,IF_size - padding_size):
			for k in range(padding_size,IF_size - padding_size):
				if (random_number < new_sparsity):
					all_zeros_IF_map[i][j][k] = 1
				random_number = random.randint(0,99)

	return all_zeros_IF_map

def sparsify_weight_map(weight_kernel_number, channel_size, weight_size, weight_sparsity):
	"""
	Same as IF map sparsification, but without considering padding problem
	"""

	all_zeros_weight_map = construct_weight_map(weight_kernel_number, channel_size, weight_size)

	random_number = random.randint(0,99)

	for i in range(weight_kernel_number):
		for j in range(channel_size):
			for k in range(weight_size):
				for l in range(weight_size):
					if (random_number < weight_sparsity):
						all_zeros_weight_map[i][j][k][l] = 1
					random_number = random.randint(0,99)

	return all_zeros_weight_map

def 



multi_sparsity_IF = [[[SRAM(0,0,0,1) for i in range(SIZE_2DIF)] for j in range(SIZE_2DIF)]for k in range(CHANNEL_2DIF)]
multi_sparsity_W = [[SRAM(0,0,0,1) for i in range(SIZE_2DW)] for j in range(CHANNEL_2DW)]



if __name__ == '__main__':
	#Constant
	IF_CHANNEL = 3
	raw_IF_size = 227

	weight_kernel_number = 96
	channel_size = 3
	weight_size = 11
	padding_size = 0
	stride = 4

	IF_sparsity = 98
	weight_sparsity = 87

	IF_size = apply_padding(raw_IF_size,padding_size)
	IF_map = sparsify_IF_map(IF_size, raw_IF_size, channel_size, IF_sparsity, padding_size)
	weight_map = sparsify_weight_map(weight_kernel_number, channel_size, weight_size, weight_sparsity)





