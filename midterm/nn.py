import random
from micrograd.engine import Value

class Module:#定義一個模組

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):#一個神經元為一個Module 

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):#x表示這個值被帶入Neuron的時候會得到的值 (是一個正向運算) #計算逆向時只需要呼叫backward()
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)#對每一個w，x做一次wi*xi最後加上b
        return act.relu() if self.nonlin else act#將對每一個w，x做一次wi*xi最後加上b算出來的結果relu，就是一個神經元的輸出值

    def parameters(self):
        return self.w + [self.b]#self.w為一個陣列，python陣列相加是append把w跟b合起來成為一個陣列

    def __repr__(self):#轉字串
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module): #神經元做完換做layer

    def __init__(self, nin, nout, **kwargs):#nin個輸入 ，nout個輸出
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)] #一個Layer是一堆神經元

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]#對於所有的neurons神經元，把所有的參數集合起來

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):#多層感知器

    def __init__(self, nin, nouts):
        sz = [nin] + nouts #每一層的大小包含輸入層，中間層，輸出層
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]#呼叫layer把每一層建立起來，每一層就是這一層的數量接到下一層的數量

    def __call__(self, x):
        for layer in self.layers: #一層一層的帶進去x後再得到輸出
            x = layer(x)
        return x#得到總輸出

    def parameters(self):#取得全部的w集合起來變成一個陣列
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):#印出
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
