import numpy as np
import random
import yaml
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


def read_file_content(file_name):
	"""

	This function reads the content line by line and stores the result in a list

	- Input:
		* parent_directory: Since state_files and promotions files are not kept in the same
			directory, we need to specify the location (ex. 'state_txt/')
		* file_name: name of file, may be promotion_key (ex. machine_state)

	- Return:
		* content_list: list of read-in data from text file

	"""

	with open (file_name, encoding = "utf8") as read_file:
		content = yaml.load(read_file,Loader=yaml.FullLoader)

	print ("Content = ",content)
	return content


#Class for compress data
class DataPoint():
	"""
	This is the data point for IF and weight 
	Each takes up to 1 byte
	"""
	def __init__(self):
		self.in_SRAM = 0
		self.size = 0
		self.flag = None
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

def compute_window_slide_number(IF_size, PE_size):
	"""
	This function computes the number of sliding weight blocks 
	all over the input feature map to cover all computations
	The number is around IF_size / weight_size
	This should also match the output feature size 
	Apply the (N+2P-F)/S + 1 formula
	"""
	# Case for weight fits in PE unit
	if weight_size < PE_size:
		window_number = (IF_size) / PE_size + 1
	else:
		raise Exception("Weight size is greater than PE size, need another algorithm")

	return int(window_number)


def construct_IF_points(window_number, channel_size):
	"""
	This constructs the mapping for IF data
	"""
	# print ("window_number = ",window_number)

	IF_data = [[[DataPoint() for i in range(window_number)] for j in range(window_number)] for k in range(channel_size)]

	return IF_data

def construct_weight_points():
	"""
	This constructs the mapping for weight points 
	"""
	
	pass


def count_IF_sparsity(IF_map, IF_data, channel_size, window_number, weight_size, sparsity_threshold, stride):
	"""
	Given the left-most and upper-most point of an IF block,
	compute the sparsity count 
	"""

	weight_size_squared = window_number * window_number

	print ("window_number = ",window_number)

	# print ("weight_size = ",weight_size)

	for i in range(channel_size):
		for j in range(window_number):
			row_index = stride * j
			for k in range(window_number):
				column_index = stride * k
				count = 0
				for p in range(window_number):
					sub_row_index = row_index + p
					for q in range(window_number):
						sub_column_index = column_index + q
						count += IF_map[i][sub_row_index][sub_column_index]

				# print("Count = ",count)
				if (count < sparsity_threshold * weight_size_squared):
					IF_data[i][j][k].flag = "sparse"
					IF_data[i][j][k].size = count

				else:
					IF_data[i][j][k].flag = "dense"
					IF_data[i][j][k].size = weight_size_squared 
	return IF_data


def compute_IF_linked_list(IF_linked_list, IF_data, weight_kernel_number, window_number, channel_size, PE_number, IF_SRAM_size):
	"""
	This gives the IF DRAM access number
	"""

	IF_DRAM_access, PSum_DRAM_access = 0, 0 

	IF_SRAM_current_data_size = 0

	kernel_loop_number = weight_kernel_number // PE_number

	for kernel in range(kernel_loop_number):
		for j in range(window_number):
			for k in range(window_number):
				for i in range(channel_size):
					fetched_block = IF_data[i][j][k]
					if (fetched_block.in_SRAM == 0):
						# print ("fetched_block size = ",fetched_block.size)
						IF_DRAM_access += fetched_block.size
						IF_SRAM_current_data_size += fetched_block.size
						IF_linked_list.add_list_item(fetched_block)

						while (IF_SRAM_current_data_size > IF_SRAM_size):
							IF_SRAM_current_data_size -= IF_linked_list.head.size
							IF_linked_list.remove_head()
							

	return IF_DRAM_access


# multi_sparsity_IF = [[[SRAM(0,0,0,1) for i in range(SIZE_2DIF)] for j in range(SIZE_2DIF)]for k in range(CHANNEL_2DIF)]
# multi_sparsity_W = [[SRAM(0,0,0,1) for i in range(SIZE_2DW)] for j in range(CHANNEL_2DW)]


if __name__ == '__main__':
	#Constant
	Loaded_content = read_file_content("../param/AlexNet_param.yaml")["Layer_one"]

	IF_CHANNEL, raw_IF_size = Loaded_content["IF_CHANNEL"], Loaded_content["raw_IF_size"]
	weight_kernel_number = Loaded_content["weight_kernel_number"]
	channel_size, weight_size = Loaded_content["channel_size"], Loaded_content["weight_size"]
	padding_size, stride = Loaded_content["padding_size"], Loaded_content["stride"]

	IF_sparsity, weight_sparsity = Loaded_content["IF_sparsity"], Loaded_content["weight_sparsity"]
	sparsity_threshold = Loaded_content["sparsity_threshold"]
	PE_size, PE_number = Loaded_content["PE_size"], Loaded_content["PE_number"]

	weight_SRAM_size, IF_SRAM_size = Loaded_content["weight_SRAM_size"], Loaded_content["IF_SRAM_size"]


	IF_size = apply_padding(raw_IF_size,padding_size)
	IF_map = sparsify_IF_map(IF_size, raw_IF_size, channel_size, IF_sparsity, padding_size)
	weight_map = sparsify_weight_map(weight_kernel_number, channel_size, weight_size, weight_sparsity)

	# Number of 2D slides
	window_number = compute_window_slide_number(IF_size, PE_size)

	IF_data = construct_IF_points(window_number, channel_size)
	IF_data = count_IF_sparsity(IF_map, IF_data, channel_size, window_number, weight_size, sparsity_threshold, stride)

	IF_linked_list = Linkedlist()

	IF_DRAM_access = compute_IF_linked_list(IF_linked_list, IF_data, weight_kernel_number, window_number, channel_size, PE_number, IF_SRAM_size)

	print ("DRAM access = ",IF_DRAM_access)



