import numpy as np 

f = open('raw_data.txt', "r")
data = np.zeros((86,2))
index = 0
for line in f:
	data[index][0] = line[7:13]
	data[index][1] = line[20:-1]
	index += 1
np.savetxt("data.csv", data)