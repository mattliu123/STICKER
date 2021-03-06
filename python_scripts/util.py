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
	def __init__(self, in_SRAM, least_recently_used, size, first_in_SRAM):
		self.in_SRAM = 0
		self.least_recently_used = 0
		self.size = 0
		self.first_in_SRAM = 1
		self.next = None
#Class for SRAM List
class Linkedlist():
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
		


#Variable
randnum = random.randint(0,99)
count = 0
W_DRAM_access = 0
IF_DRAM_access = 0
IF_SRAM_num = 0
W_SRAM_num = 0
PSUM_SRAM_num = 0

#temp
temp_use = 0
temp_i = 0
temp_j = 0
temp_k = 0



#Constant
IF_CHANNEL = 3
IF_SIZE = 227

W_KERNEL = 96
W_CHANNEL = 3
W_SIZE = 11

CHANNEL_2DIF = IF_CHANNEL
SIZE_2DIF = IF_SIZE//16 + 1

CHANNEL_2DW = W_CHANNEL
SIZE_2DW = W_KERNEL

PADDING = 0
STRIDE = 4
#Layer0 ifmap: 227*227*3 filter 11*11*3(96 kernels)

SPARTSITY_THRESHOLD = 0.5

Input_fmap = np.zeros((IF_CHANNEL, IF_SIZE + 13, IF_SIZE + 13), dtype = int)
filter_map = np.zeros((W_KERNEL, W_CHANNEL, W_SIZE, W_SIZE), dtype = int)

multi_sparsity_IF = []
multi_sparsity_W = []


for i in range(IF_CHANNEL):
	for j in range(IF_SIZE):
		for k in range(IF_SIZE):
			if(randnum < 98):
				Input_fmap[i][j][k] = 1
			randnum = random.randint(0,99)

for i in range(W_KERNEL):
	for j in range(W_CHANNEL):
		for k in range(W_SIZE):
			for l in range(W_SIZE):
				if(randnum < 87):
					filter_map[i][j][k][l] = 1
				randnum = random.randint(0,99)
multi_sparsity_IF = [[[SRAM(0,0,0,1) for i in range(SIZE_2DIF)] for j in range(SIZE_2DIF)]for k in range(CHANNEL_2DIF)]
multi_sparsity_W = [[SRAM(0,0,0,1) for i in range(SIZE_2DW)] for j in range(CHANNEL_2DW)]

'''

Multisparsity mode

'''
WEIHT_SRAM_list = Linkedlist()
IF_SRAM_list = Linkedlist()

for i in range(IF_CHANNEL):
	for j in range(SIZE_2DIF):
		for k in range(SIZE_2DIF):
			count = 0
			for a in range(16):
				for b in range(16):
					count = count + Input_fmap[i][16*j+a][16*k+b]
					
			print ("count = ",count)

			if(count<SPARTSITY_THRESHOLD*256):
				multi_sparsity_IF[i][j][k].size = count
			else:
				multi_sparsity_IF[i][j][k].size = 256

for i in range(CHANNEL_2DW):
	for j in range(SIZE_2DW):
		count = 0
		for a in range(W_SIZE):
			for b in range(W_SIZE):
				count = count + filter_map[j][i][a][b]
		if(count<61):
			multi_sparsity_W[i][j].size = count
		else:
			multi_sparsity_W[i][j].size = 121

'''
PE ARRAY: Channel -> 2D -> Kernel

'''

for t in range(W_KERNEL//16):
	for j in range(SIZE_2DIF):
		for k in range(SIZE_2DIF):
			for i in range(CHANNEL_2DIF):
				data = multi_sparsity_IF[i][j][k]
				if(data.in_SRAM == 0):
					IF_SRAM_num = IF_SRAM_num + data.size
					IF_DRAM_access = IF_DRAM_access + data.size
					IF_SRAM_list.add_list_item(data)
					while(IF_SRAM_num > 16000):
						IF_SRAM_num -= IF_SRAM_list.head.size
						IF_SRAM_list.remove_head()

#Weight DRAM access

for t in range(6):
	for k in range(225):
		for i in range(CHANNEL_2DIF):
			for j in range(16):
				weight_data = multi_sparsity_W[i][16*t+j]
				if(weight_data.in_SRAM == 0):
					W_SRAM_num = W_SRAM_num + weight_data.size
					W_DRAM_access = W_DRAM_access + weight_data.size
					WEIHT_SRAM_list.add_list_item(weight_data)
					while(W_SRAM_num > 16000):
						W_SRAM_num -= WEIHT_SRAM_list.head.size
						WEIHT_SRAM_list.remove_head()  
						
print("IF DRAM Access data for Layer1 = ",IF_DRAM_access)
print("Weight DRAM Access data for Layer1 = ",W_DRAM_access)

