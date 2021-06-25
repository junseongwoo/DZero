# Step_22 연산자 오버로드(3)
import numpy as np 
import unittest
import weakref

class Variable:
    __array_priority__ = 200
    def __init__(self, data, name=None):
        # ndarray 가 아닌 데이터 타입이 오면 에러 띄움 
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None 
        self.generation = 0 # 세대 수를 기록하는 변수

    def set_creator(self, func): 
        self.creator = func
        self.generation =func.generation + 1 # 세대를 기록 (부모 세대 + 1)
    
    def backward(self, retain_grad=False):
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
            gys = [output().grad for output in f.outputs]
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
            
            if not retain_grad :
                for y in f.outputs:
                    y().grad = None # y는 약한 참조 
    # step_14 
    def cleargrad(self):
        self.grad = None
    
    # step_19 
    # @property를 사용하여 shape()가 아닌 self.shape 처럼 변수로 사용 가능
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None : 
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'
    # * 연산자 오버로딩 
    def __mul__(self, other):
        return mul(self, other)
    # 더 간단하게 만들수 있음 
    #Variable.__mul__ = mul 

class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs] # Step_21 

        xs = [x.data for x in inputs]
        ys = self.forward(*xs) 
        if not isinstance(ys, tuple): 
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

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
    # Step_21 : x1이 float, int 경우 ndarray로 반환하여 계산
    x1 = as_array(x1)
    return Add()(x0, x1)

#Step_18 : 모드 전환을 위한 클래스
class Config:
    enable_backprop = True

#Step_20 : * 연산자 지원 
class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0 

# Mul 클래스를 파이썬 함수로 사용하게 만드는 작업 
def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)

# Step_21 
def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

# 파이썬에서는 함수도 객체이므로 함수 자체를 할당 할 수 있다.
Variable.__add__ = add
Variable.__radd__ = add 
Variable.__mul__ = mul 
Variable.__rmul__ = mul 

# step_21 부호 변환
class Neg(Function):
    def forward(self, x):
        return -x
    def backward(self, grad):
        return -grad

def neg(x):
    return Neg()(x)

Variable.__neg__ = neg

# step_21 뺄셈
class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1 
        return y
    def backward(self, gy):
        return gy, -gy

def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

Variable.__sub__ = sub

def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)

Variable.__rsub__ = rsub

class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1
def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

Variable__truediv__ = div
Variable__rtruediv__ = rdiv

class Pow(Function):
    def __init__(self, c):
        self.c = c 

    def forward(self, x):
        y = x ** self.c
        return y 
    
    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c 
        gx = c * x ** (c-1) * gy
        return gx

def pow(x, c):
    return Pow(c)(x)

Variable.__pow__ = pow 

if __name__ == "__main__":
    x = Variable(np.array(2.0))
    y = 0.26 * x
    print(y)