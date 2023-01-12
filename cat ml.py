import numpy as np
import matplotlib.pyplot as plt
import math
datap = np.random.rand(1000,2) * 30
print(datap)
X,Y = np.split(datap,2,axis=1)
plt.plot(X,Y)
mean_x = np.mean(X)
mean_y = np.mean(Y)
m = len(X)
numer = 0
denom = 0
for i in range(m):
  numer += (X[i] - mean_x) * (Y[i] - mean_y)
  denom += (X[i] - mean_x) ** 2
m = numer / denom
c = mean_y - (m * mean_x)
print (f'm = {m} \nc = {c}')
max_x = np.max(X) + 100
min_x = np.min(Y) - 100
x = np.linspace (min_x, max_x, 100)
y = c + m * x
plt.plot(x, y, color='#58b970', label='Regression Line')
plt.scatter(X, Y, c='#ef5423', label='data points')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
ss_t = 0 
ss_r = 0 

for i in range(int(120)): 
  y_pred = c + m * X[i]
  ss_t += (Y[i] - mean_y) ** 2
  ss_r += (Y[i] - y_pred) ** 2
r2 = 1 - (ss_r/ss_t)

print(r2)
sum1=0
diff=0
for i in range(int(m)):
    y_pred = c + m * X[i]
    diff=abs(y_pred-Y[i])
    sum1+=diff
total=sum1/m
total
sum1=0
diff1=0
for i in range(int(m)):
    y_pred = c + m * X[i]
    diff1 = (y_pred-Y[i]) ** 2
    sum1 += diff1    
total1=math.sqrt(sum1/m)
total1