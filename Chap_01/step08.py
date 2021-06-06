## Step_8 : 재귀에서 반복문으로 

import numpy as np 

class Variable:
    def __init__(self, data) -> None:
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):  # New
        self.creator = func
    '''
    def backward(self):
        f = self.creator                   ## 1. 함수를 가져온다. 
        if self.creator is not None:
            x = f.input                    ## 2. 함수의 입력을 가져온다.
            x.grad = f.backward(self.grad) ## 3. 함수의 backward 메서드를 호출한다.
            x.backward()
    '''

    ## 위의 방법은 재귀를 이용했지만
    ## 반복문을 이용해 backward를 구현해본다.
    def backward(self):
        funcs = [self.creator]
        while funcs :
            f = funcs.pop()                ## 함수를 가져옴 
            x, y = f.input, f.output       ## 함수의 입력과 출력을 가져옴
            x.grad = f.backward(y.grad)    ## backward 함수 호출 

            if x.creator is not None :
                funcs.append(x.creator)    ## 하나 앞의 함수를 리스트에 추가

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

if __name__ == "__main__":
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)

