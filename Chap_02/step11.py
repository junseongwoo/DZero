# Step_11 : 가변 길이 인수 (순전파)
# 이때까지 하나의 값을 받아 하나의 출력을 냈지만
# Function을 수정해 가변 길이의 입출력을 받으려함 
#   - 변수들을 리스트 or 튜플에 넣어 처리 
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
    # __call__ 메서드 
    # 1. Variable라는 상자에서 실제 데이터를 꺼내고 
    # 2. forward 메서드에서 구체적인 계산 실시
    # 3. 계산 결과를 Variable에 넣고
    # 4. 자신이 '창조자' 라고 원산지 표시
    def __call__(self, inputs):
        # 리스트 내포 : inputs 리스트의 각 원소 x에 대해 
        # 각각의 데이터 (x.data)를 꺼내고, 꺼낸 원소들로 새로운 리스트를 만듬
        xs = [x.data for x in inputs]
        ys = self.forward(xs)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        
        self.inputs = inputs
        self.output = outputs
        return outputs

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

''' 
파이썬에서 square과 exp 함수를 제공 
클래스의 인스턴스로 생성한 다음 인스턴스를 호출해야하는 단계를 하나로 줄인다.
'''
def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def as_array(x):
    ## isscalar : 입력 데이터가 numpy.float64 같은 스칼라 타입인지 확인해주는 함수
    ##    - True or False 
    if np.isscalar(x):
        return np.array(x)
    return x

# step10-1 파이썬 단위 테스트
class SquareTest(unittest.TestCase):
    # 테스트 규칙 : 이름이 test로 시작하게 만들고 그 안에 테스트할 내용 적음 
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected) # 함수의 출력이 기댓값과 같은지 확인하는 것 

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1)) # 무작위 입력값 생성 
        y = square(x)
        y.backward() # 역전파로 미분 값 구함
        num_grad = numerical_diff(square, x) # 수치 미분으로도 구해본다.
        flg = np.allclose(x.grad, num_grad)  # 구한 값들이 거의 일치하는지 확인하는 np.allclose
        self.assertTrue(flg)

#Step_11 
class Add(Function):
    def forward(self, xs):
        x0, x1 = xs
        y = x0 + x1
        return (y,)

if __name__ == "__main__":
    xs = [Variable(np.array(2)), Variable(np.array(3))]
    f = Add()
    ys = f(xs)

    y = ys[0]
    print(y.data)

   

