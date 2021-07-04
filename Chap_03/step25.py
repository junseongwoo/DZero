# step_25 ~ 26 : 게산 그래프 시각화(1)

# 계산 그래프를 시각화 하면 문제가 발생했을 때 원인 파악 쉬워짐 
# 신경망의 구조를 3자에게 시각적으로 전달하는 용도로도 활용

# 역전파는 출력 변수를 기점으로 역방향으로 모든 노드 (변수와 함수)를 추적 
#  - 이것을 활용하여 계산 그래프의 노드를 DOT 언어로 변환 

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph

def goldstein(x,y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*(x**2) - 14*y + 6*x*y + 3*(y**2))) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*(x**2) + 48*y - 36*x*y + 27*(y**2)))
    return z

x = Variable(np.array(1.0))
y = Variable(np.array(1.0)) 
z = goldstein(x,y)
z.backward()

x.name='x'
y.name='y'
z.name='z'
plot_dot_graph(z, verbose=False, to_file='goldstein.png')
