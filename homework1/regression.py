import matplotlib.pyplot as plt
import numpy as np
import random

#x = np.array([0, 1, 2, 3, 4], dtype=np.float32)
#y = np.array([2, 3, 4, 5, 6], dtype=np.float32)
x = np.array([0, 1, 2, 3, 4], dtype=np.float32)
y = np.array([1.9, 3.1, 3.9, 5.0, 6.2], dtype=np.float32)

def predict(a, xt):
	return a[0]+a[1]*xt 

def MSE(a, x, y):
	total = 0
	for i in range(len(x)):
		total += (y[i]-predict(a,x[i]))**2
	return total

def loss(p):
	return MSE(p, x, y)

# p = [0.0, 0.0]
# plearn = optimize(loss, p, max_loops=3000, dump_period=1)

# 請修改這個函數，自動找出讓 loss 最小的 p
# 這個值目前是手動填的，請改為自動尋找。(即使改了 x,y 仍然能找到最適合的回歸線)

def optimize(p , loss ,h =0.01):
	failcount = 0
	while(failcount<=10000):
		dp1 = random.uniform(-h,h)
		dp2 = random.uniform(-h,h)
		dp = [p[0]+dp1,p[1]+dp2]
		if( loss(dp) <= loss(p)):
			p[0] = p[0] +dp1
			p[1] = p[1] +dp2
			failcount =0
		else:
			failcount = failcount+1

	return p
p = optimize([0.0,0.0],loss)

# Plot the graph
y_predicted = list(map(lambda t: p[0]+p[1]*t, x))
print('y_predicted=', y_predicted)
plt.plot(x, y, 'ro', label='Original data')
plt.plot(x, y_predicted, label='Fitted line')
plt.legend()
plt.show()