
class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other): # f=x+y
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad  #  因為 f=x+y  gx微分是1 偏微分時直接乘1回來 => gx = gf，一定要使用加等於因為有可能一個節點輸出給不同個再去算出結果，權重會累加
            other.grad += out.grad # 同理 gy = gf
        out._backward = _backward

        return out

    def __mul__(self, other): # f=x*y
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*') 

        def _backward():
            self.grad += other.data * out.grad # gx = y*gf  假設f的梯度值是gf x的梯度值是 y*gf   gx/gf = y
            other.grad += self.data * out.grad # gy = x*gf  
        out._backward = _backward

        return out

    def __pow__(self, other): # f = x**n x的n次方
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}') #other次方是個常數

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad # gx = n (x**n-1)  n乘上x的n-1次方
        out._backward = _backward

        return out

    def relu(self): #(以正向和負向的結果當作條件)
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU') #正向小於0的時候就等於0

        def _backward(): 
            self.grad += (out.data > 0) * out.grad # 如果f>0 gx=1
        out._backward = _backward

        return out

    def backward(self): #當最後的正向值算出來後要一層層的往回算

        # topological order all of the children in the graph #計算順序因為要逆向回去，否則算不出來的元素先算會有問題
        topo = []#
        visited = set()#類似dfs 紀錄拜訪過的
        def build_topo(v):
            if v not in visited: #找出未被拜訪過的加入visited.add
                visited.add(v)
                for child in v._prev:#如果v 在child前一個 把他加進去
                    build_topo(child)
                topo.append(v)#topological 順序算出來後再放入topo陣列中
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):#將topo倒過來
            v._backward()#算一遍後結束

    def __neg__(self): # 取-self的值
        return self * -1

    def __radd__(self, other): # other + self 加等於 +=
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other 
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self): # 轉字串 -- https://www.educative.io/edpresso/what-is-the-repr-method-in-python
        return f"Value(data={self.data}, grad={self.grad})"
