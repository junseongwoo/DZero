# Step_24 : 복잡한 함수의 미분 
# 여기서 다루는 함수는 최적화에 자주 사용되는 테스트 함수이다.
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable

def sphere(x,y):
    z = x ** 2 + y ** 2 
    return z 

def matyas(x,y):
    z = 0.26 * (x**2 + y**2) - 0.48 * x * y
    return z 

x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
#z = sphere(x,y)
z = matyas(x,y)
z.backward()
print(x.grad, y.grad )