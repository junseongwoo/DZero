# Step_9 : 함수를 더 편리하게 
# 3가지 개선
# 1. 파이썬 함수 이용 line 66
# 2. backward 메서드 간소화 
# 3. ndarray만 취급 
import numpy as np 

class Variable:
    def __init__(self, data) -> None:
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
        output = Variable(y)
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

    def backward(self, grad):  # grad from prev step
        x = self.input.data
        grad = (2 * x) * grad  # grad of current step
        return grad

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, grad):
        x = self.input.data
        grad = np.exp(x) * grad
        return grad

''' 
파이썬에서 square과 exp 함수를 제공 
클래스의 인스턴스로 생성한 다음 인스턴스를 호출해야하는 단계를 하나로 줄인다.
'''
def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

if __name__ == "__main__":

    '''
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)
    '''

    x = Variable(np.array(0.5))
    '''
    a = square(x)
    b = exp(a)
    y = square(b)
    '''

    y = square(exp(square(x)))
    #y.grad = np.array(1.0)
    y.backward()
    print(x.grad)

