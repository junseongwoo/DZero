# Step_10 : 테스트 
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
        # y.grad = np.array(1.0)를 생략하기 위해 
        # 변수의 grad가 None 이면 자동으로 미분값을 생성하게 함
        # np.ones_like(self.data)는 self.data와 형상과 데이터 타입이 같은 ndarray 인스턴스 생성
        #   - 모든 요소를 1로 채워서 돌려줌 
        #   - self.data가 스칼라이면 self.grad도 스칼라가 됌 
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
    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)  # New
        self.input = input
        self.output = output  # New
        return output

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
# 
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


if __name__ == "__main__":

    # x = Variable(np.array([0.5])) ## 1차 // 실행 됌
    x = Variable(np.array(0.5))     ## 0차 // 0차 ndarray를 계산하면 y가 float64가 된다 
    
    y = square(exp(square(x)))
    #y.grad = np.array(1.0)
    y.backward()
    print(x.grad)

   

