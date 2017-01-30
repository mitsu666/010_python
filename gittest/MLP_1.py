import numpy as np
from numpy.random import *
from  sklearn.datasets import *


mnist = fetch_mldata('MNIST original', data_home=".")
p = random_integers(0, len(mnist.data), 1)

data=mnist['data']
target=mnist['target']
x_train,x_test = np.split(data,[60000])
y_train,y_test = np.split(target,[60000])

#print(x_train[1])
def distance(p1, p2):
    # N 次元 (配列) で 2 点間のユークリッド距離を求める
    return np.sum((p1 - p2) ** 2)


def nearest_neighbor(x_train, x_target, point):
    # point と train の各点のユークリッド距離を測る
    distances = np.array([distance(t, point) for t in x_train])
    # 距離が最小 (再近傍) の点を得る
    nearest_point = distances.argmin()
    # 再近傍の点の種別を判定結果として返す
    return x_target[nearest_point]
y_test1=y_test[1:1001]
x_test1=x_test[1:1001]
#predict=nearest_neighbor(x_train,y_train,x_test[3456])
#print (type(y_test))
results=[]
for l,k in zip(x_test1,y_test1):
    #print (i)
    predict=nearest_neighbor(x_train,y_train,l)
    #print (predict)
    #print (k) 
    results.append(predict==k)
print (results)
N=len(y_test1)
correct = results.count(True)
print (float(correct/N))
#print (predict)
#print (y_test[3456])
#print(data[1])
#print(target[1])

#for i,(data,label) in enumerate(np.array(zip(mnist.data,mnist.target))[p]):
#    print (label)


#print (p)
