import os, sys
sys.path.append(os.pardir)
from common.functions import *
from common import gradient

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01) -> None:
        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)

        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
    def predict(self, x):
        """
        学習済みパラメータで予測を行う
        """
        W1, W2 = self.params['W1'], self.params['W2'] 
        b1, b2 = self.params['b1'], self.params['b2'] 
        
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
    
    def loss(self, x, t):
        """
        交差エントロピー誤差を計算

        Args:
            x 入力データ
            t 教師データ
        Returns:
            交差エントロピー誤差
        """
        y = self.predict(x)
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        acurracy = np.sum(y==t) / float(x.shape[0])
        return acurracy
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x,t)
        
        grads = {}
        grads['W1'] = gradient.numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = gradient.numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = gradient.numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = gradient.numerical_gradient(loss_W, self.params['b2'])
        
        return grads

