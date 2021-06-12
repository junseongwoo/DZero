# Step_13 가변 길이 인수 (역전파)

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

    def set_creator(self, func): 
        self.creator = func
    
    def backward(self):
        if self.grad is None: 
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs :
            f = funcs.pop()                
            x, y = f.input, f.output       
            x.grad = f.backward(y.grad)    

            if x.creator is not None :
                funcs.append(x.creator)    

class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs) 
        if not isinstance(ys, tuple): 
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        
        self.inputs = inputs
        self.output = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, grad):  
        x = self.input.data
        grad = (2 * x) * grad  
        return grad

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

#Step_12 : Add 클래스를 파이썬 함수로 만듬
def add(x0, x1):
    return Add()(x0, x1)

if __name__ == "__main__":
    x0 = Variable(np.array(2))
    x1 = Variable(np.array(3))

    y = add(x0, x1) 
    print(y.data)
