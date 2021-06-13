# Step_16 복잡한 계산 그래프 (구현)

import numpy as np 
import unittest 

class Variable:
    def __init__(self, data) -> None:
        # ndarray 가 아닌 데이타 타입이 오면 에러 띄움 

        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0 # 세대 수를 기록하는 변수

    def set_creator(self, func): 
        self.creator = func
        self.generation =func.generation + 1 # 세대를 기록 (부모 세대 + 1)
    
    def backward(self):
        if self.grad is None: 
            self.grad = np.ones_like(self.data)
            
        # Step_14
        funcs = [] 
        seen_set = set() 

        def add_func(f):
            if f not in seen_set :
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x : x.generation)
        add_func(self.creator)

        while funcs :
            f = funcs.pop()        
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )

            for x, gx in zip(f.inputs, gxs):
                # Step_14
                if x.grad is None :
                    x.grad = gx 
                else :
                    x.grad = x.grad + gx

                if x.creator is not None :
                    #funcs.append(x.creator)    
                    add_func(x.creator) # step_14 
    # step_14 
    def cleargrad(self):
        self.grad = None
                    

class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs) 
        if not isinstance(ys, tuple): 
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)
        
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):  
        #Step_13 
        #x = self.input.data
        x = self.inputs[0].data 
        gy = (2 * x) * gy  
        return gy

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, grad):
        x = self.input.data
        grad = np.exp(x) * grad
        return grad

# 기울기를 구하는 함수 
def numerical_diff(f, x, eps = 1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2*eps)

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

# step10-1 파이썬 단위 테스트
class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1)) 
        y = square(x)
        y.backward() 
        num_grad = numerical_diff(square, x) 
        flg = np.allclose(x.grad, num_grad)  
        self.assertTrue(flg)

#Step_11 
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy

#Step_12 : Add 클래스를 파이썬 함수로 만듬
def add(x0, x1):
    return Add()(x0, x1)

if __name__ == "__main__":
    x = Variable(np.array(2.0))
    
    a = square(x)
    y = add(square(a), square(a))
    y.backward()

    print(y.data)
    print(x.grad)