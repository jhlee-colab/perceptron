# Perceptron & DNN

생성자: 이종훈
생성 일시: July 1, 2025 2:06 PM
카테고리: DNN
최종 편집자:: 이종훈
최종 업데이트 시간: August 19, 2025 2:31 PM

# Deep Learning

## Why Deep Learning?

데이터에서 직접 특징들을 배울 수 있을까?

- 수작업으로 얻어진 특징들
    - 많은 시간이 소요
    - 깨지기 쉬움
    - 확장이 용이하지 않음
- Features
    - Low Level Features
        
        ![Lines & Edges](Perceptron%20&%20DNN%2022327edcc17f803bb43cc5bdf11faddd/e2ce9bf6-1ae2-4c39-90b5-14c441aa8883.png)
        
        Lines & Edges
        
    - Mid Level Features
        
        ![Eyes & Nose & Ears](Perceptron%20&%20DNN%2022327edcc17f803bb43cc5bdf11faddd/35a2eaa2-2f5b-426e-b010-64601f421c75.png)
        
        Eyes & Nose & Ears
        
    - High Level Features
        
        ![Facial Structure](Perceptron%20&%20DNN%2022327edcc17f803bb43cc5bdf11faddd/ee61e95c-937b-4ba7-9a08-045292a0a74b.png)
        
        Facial Structure
        

## The Perceptron

### Forward Propagation

- Output
    
    $\hat y = g(\sum^m_{i=1}{x_iw_i})$
    
    - Linear combination of inputs: $\sum^m_{i=1}{x_iw_i}$
    - Non-linear activation function: $g(\cdot)$
    

![mit.png](Perceptron%20&%20DNN%2022327edcc17f803bb43cc5bdf11faddd/mit.png)

- Output with bias
    
    $\hat y=g(w_0 + \sum^m_{i=1}{x_iw_i})$
    
    $\hat y=g(w_0 + \mathbf{x^T}\mathbf{w})$
    
    - Linear combination of inputs: $w_0+\sum^m_{i=1}{x_iw_i}$
    - Non-linear activation function: $g(\cdot)$
    - $\mathbf{x}=\begin{bmatrix} x_1 \\ \vdots \\ x_m \end{bmatrix}$ and $\mathbf{w}=\begin{bmatrix} w_1 \\ \vdots \\ w_m \end{bmatrix}$

![mit.png](Perceptron%20&%20DNN%2022327edcc17f803bb43cc5bdf11faddd/mit%201.png)

### Activation Functions

네트워크에 비선형성을 적용하기 위해서 활용되는 함수들

- Sigmoid
    - $f(z)=\frac{1}{1+e^{-z}}$
    - $f'(z)=f(z)\{1-f(z)\}$
    
    ![download.png](Perceptron%20&%20DNN%2022327edcc17f803bb43cc5bdf11faddd/download.png)
    
- Hyperbolic Tangent
    - $f(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}$
    - $f'(z)=1-f(z)^2$
    
    ![download-1.png](Perceptron%20&%20DNN%2022327edcc17f803bb43cc5bdf11faddd/download-1.png)
    
- Rectified Linear Unit(ReLU)
    - $f(z)=\max(0, z)$
    - $f'(z)=\begin{cases} 1, & z \gt 0 \\ 0, & \text{otherwise} \end{cases}$
    
    ![download-2.png](Perceptron%20&%20DNN%2022327edcc17f803bb43cc5bdf11faddd/download-2.png)
    

<aside>
💡

**[Example]**

when $w_0=1$ and $\mathbf{w}=\begin{bmatrix}3 \\ -2 \end{bmatrix}$

$\hat y = g(w_0+\mathbf{x^T}\mathbf{w})=g(1+\begin{bmatrix}x_1 \\ x_2 \end{bmatrix}^T \begin{bmatrix}3 \\ -2 \end{bmatrix})$

$\hat y = g(1+3x_1-2x_2)$

1) 입력 $\mathbf{x}=\begin{bmatrix}-1 \\ 2 \end{bmatrix}$

$\hat y = g(1 + (3 \times-1)-(2\times 2))=g(-6)\approx0.002$

2) 입력 $\mathbf{x}=\begin{bmatrix}1 \\ -2 \end{bmatrix}$

$\hat y = g(1 + (3 \times1)-(2\times -2))=g(8)\approx0.9997$

3) $z=0$

$\hat y = g(0)=\frac{1}{1+e^{-0}}=\frac{1}{1+1}=0.5$

![mit.png](Perceptron%20&%20DNN%2022327edcc17f803bb43cc5bdf11faddd/f09ecd37-32b6-412c-8daa-7b8c0e8c8742.png)

![download-4.png](Perceptron%20&%20DNN%2022327edcc17f803bb43cc5bdf11faddd/download-4.png)

</aside>

### The Perceptron: Simplified

![mit.png](Perceptron%20&%20DNN%2022327edcc17f803bb43cc5bdf11faddd/cf0dc832-f216-4807-a01b-9f021262f6fe.png)

$z=w_0+\sum^m_{j=1}x_jw_j$

$y=g(w_0+\sum^m_{j=1}x_jw_j)$

### Multi-Output Perceptron

- Dense layers: 모든 입력이 모든 출력과 조밀하게 연결되어 있어 dense layer라 부른다.

![mit.png](Perceptron%20&%20DNN%2022327edcc17f803bb43cc5bdf11faddd/a723d976-e747-4efa-ae6a-211cf2246a6d.png)

$z_i=w_{0,i}+\sum^m_{j=1}x_jw_{j,i}$

$y_i=g(w_{0,i}+\sum^m_{j=1}x_jw_{j,i})$

```python
import torch
import torch.nn

class MyDenseLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bias: bool=True, act_fn: str='sigmoid') -> None:
        super().__init__()
    
        # initialize Parameters
        self.W = nn.Parameter(nn.init.xavier_normal_(torch.empty(input_dim, output_dim)), requires_grad=True)
        if bias:
            self.b = nn.Parameter(nn.init.xavier_normal_(torch.empty(1, output_dim)), requires_grad=True)
        else:
            self.b = None
        # activation function
        self.activation = self._get_activation(act_fn)

    def _get_activation(self, name: str):
        try:
            act_fn = getattr(torch, name)
        except:
            act_fn = torch.sigmoid
        return act_fn

    def reset(self):
        self.W.data.copy_(torch.randn_like(self.W.data))
        if isinstance(self.b, nn.Parameter):
            self.b.data.copy_(torch.randn_like(self.b.data))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # forward propagation
        z = inputs @ self.W + self.b if isinstance(self.b, nn.Parameter) else inputs @ self.W
        return self.activation(z)
```

### Single Layer Neural Network

- hidden layer
    
    $z_i=w^{(1)}_{0,i}+\sum^m_{j=1}x_jw_{j,i}^{(1)}$
    
- final output
    
    $\hat y_i=g(w^{(2)}_{0,i}+\sum^{d_1}_{j=1}g(z_j)w_{j,i}^{(2)})$
    

![mit.png](Perceptron%20&%20DNN%2022327edcc17f803bb43cc5bdf11faddd/7ddd32e3-2d37-4022-9641-1c4d1ea0f86e.png)

- $z_2$
    
    $z_2=w_{0,2}^{(1)}+\sum_{j=1}^m{x_jw_{j,2}^{(1)}}$
    
    $z_2=w_{0,2}^{(1)}+x_1w^{(1)}_{1,2}+x_2w^{(1)}_{2,2}+x_mw^{(1)}_{m,2}$
    

![mit.png](Perceptron%20&%20DNN%2022327edcc17f803bb43cc5bdf11faddd/ce4c45f2-12d1-4542-909c-7ae7d0f24203.png)

### Deep Neural Network

- Hidden
    
    $z_{k,i}=w^{(k)}_{0,i}+\sum_{j=1}^{n_{k-1}}{g(z_{k-1,j})w_{j,i}^{(k)}}$
    

![mit.png](Perceptron%20&%20DNN%2022327edcc17f803bb43cc5bdf11faddd/626bcf92-ab81-42cb-9d55-ab09f64680a4.png)

## Example

<aside>
💡

Will I pass this class?

simple two feature model

- $x_1$ = # of lectures you attend
- $x_2$ = Hours spent on the final project
</aside>

![image.png](Perceptron%20&%20DNN%2022327edcc17f803bb43cc5bdf11faddd/image.png)

### Quantifying Loss

- loss: 네트워크의 손실은 잘못된 예측으로 인한 비용을 측정하는 것
    
    $\mathcal{L}=(f(x^{(i)}; \mathbf{W}), y^{(i)})$
    

![image.png](Perceptron%20&%20DNN%2022327edcc17f803bb43cc5bdf11faddd/image%201.png)

### Empirical Loss

- empirical loss: 전체 데이터셋에 대한 총 손실을 측정
    - Objective function / Cost function / empirical risk
    
    $\mathcal{J}(\mathbf{W})=\frac{1}{n}\sum^n_{i=1}{\mathcal{L}(f(x^{(i)}; \mathbf{W}), y^{(i)})}$
    

![image.png](Perceptron%20&%20DNN%2022327edcc17f803bb43cc5bdf11faddd/image%202.png)

### Binary Cross Entropy Loss

- cross entropy loss: 0과 1사이의 확률을 산출하는 모델에서 사용되는 손실 함수
    
    $\mathcal{J}(\mathbf{W})=-\frac{1}{n}\sum^n_{i=1}{y^{(i)}\log(f(x^{(i)};\mathbf{W}))+(1-y^{(i)})\log(1-f(x^{(i)};\mathbf{W}))}$
    

### Mean Squared Error Loss

- MSE loss: 연속적인 실수를 산출하는 회귀 모델에서 사용되는 손실 함수
    
    $\mathcal{J}(\mathbf{W})=\frac{1}{n}\sum^n_{i=1}{(y^{(i)}-f(x^{(i)}; \mathbf{W}))^2}$
    

## Training Neural Networks

### Loss Optimization

- 가장 낮은 손실 값을 갖는 네트워크 가중치를 찾는 것
    
    $\mathbf{W}^*=\argmin\limits_{\mathbf{W}} \frac{1}{n}\sum^n_{i=1}\mathcal{L}(f(x^{(i)};\mathbf{W}), y^{(i)})$ 
    
    $\mathbf{W}^*=\argmin\limits_{\mathbf{W}} J(\mathbf{W})$ 
    
    $\rightarrow \mathbf{W}=\{\mathbf{W}^{(0)},\mathbf{W}^{(1)},\cdots\}$
    

![image.png](Perceptron%20&%20DNN%2022327edcc17f803bb43cc5bdf11faddd/image%203.png)

- loss: 네트워크 가중치의 함수 형태이다.

### Gradient Descent

- randomly pick an initial $(w_0,w_1)$
- compute gradient, $\frac{\partial J(\mathbf{W})}{\partial \mathbf{W}}$
- take small step in opposite direction of gradient
- repeat until convergence

![image.png](Perceptron%20&%20DNN%2022327edcc17f803bb43cc5bdf11faddd/image%204.png)

<aside>
💡

**[Algorithm]**

1. Initialize weights randomly $\sim \mathcal{N}(0,\sigma^2)$
2. Loop until convergence:
3.    Compute gradient, $\frac{\partial J(\mathbf{W})}{\partial \mathbf{W}}$
4.    Update weights, $\mathbf{W} \leftarrow \mathbf{W} - \eta \frac{\partial J(\mathbf{W})}{\partial \mathbf{W}}$
5. Return weights
</aside>

### Backpropagation

- 하나의 가중치(ex. $w_2$)의 작은 변화가 최종 손실에 얼만큼의 영향을 줄까?

![image.png](Perceptron%20&%20DNN%2022327edcc17f803bb43cc5bdf11faddd/image%205.png)

                              $\frac{\partial J(\mathbf{W})}{\partial w_2}=$$\frac{\partial J(\mathbf{W})}{\partial \hat y}$ x $\frac{\partial \hat y}{\partial w_2}$      ← chain rule 적용    

![image.png](Perceptron%20&%20DNN%2022327edcc17f803bb43cc5bdf11faddd/image%206.png)

                             $\frac{\partial J(\mathbf{W})}{\partial w_1}=$$\frac{\partial J(\mathbf{W})}{\partial \hat y}$ x $\frac{\partial \hat y}{\partial w_1}$

                             $\frac{\partial J(\mathbf{W})}{\partial w_1}=$$\frac{\partial J(\mathbf{W})}{\partial \hat y}$ x $\frac{\partial \hat y}{\partial z_1}$ x $\frac{\partial z_1}{\partial w_1}$      ← chain rule 적용

- 뒤로 가면서 chain rule을 적용하며, 모든 가중치에 대해 반복 수행

## Neural Network in Practice: Optimization

### Training Neural Network

- difficult
- optimize loss function
    
    $\mathbf{W}\leftarrow \mathbf{W}-\eta\frac{\partial J(\mathbf{W})}{\partial \mathbf{W}}$
    
    - $\eta$ : learning rate   ← learning rate는 어떤 값이 적당한가?
        - small learning rate: 느리게 수렴하고, local minima에 갇힐 수 있다.
        - large learning rate: 과도하여 불안정하거나 발산할 수 있다.
        - stable learning rate: 원할하게 수렴되며, local minima도 피할 수 있다.
    
    ![image.png](Perceptron%20&%20DNN%2022327edcc17f803bb43cc5bdf11faddd/image%207.png)
    

### Adaptive Learning Rates

- Idea I: 다양한 learning rate를 시도해보고, 잘 맞는 learning rate를 찾는다.
- **Idea II: 주변 상황을 고려한 adaptive learning rate를 설계한다.**
    - learning rate는 고정되어 있지 않다.
    - 고려 대상 (줄이거나 키우거나)
        - 기울기 크기
        - 학습 수렴 속도
        - 특정 가중치의 크기

### Gradient Descent Algorithms

- SGD
- Adam
- Adadelta
- Adagrad
- RMSProp

### Stochastic Gradient Descent

<aside>
💡

[Algorithm]

1. Initialize weights randomly $\sim \mathcal{N}(0, \sigma^2)$
2. Loop until convergence:
3.    pick batch of $B$ data points
4.    Compute gradient, $\frac{\partial J(\mathbf{W})}{\partial \mathbf{W}}=\frac{1}{B}\sum_{k=1}^B\frac{\partial J_k(\mathbf{W})}{\partial \mathbf{W}}$
5.    Update weights, $\mathbf{W} \leftarrow \mathbf{W} - \eta \frac{\partial J(\mathbf{W})}{\partial \mathbf{W}}$
6. Return weights
</aside>

- Mini-batch의 효과
    - 정확한 기울기 추정
    - 원활한 수렴
    - large learning rate 허용
    - 계산을 병렬화하여 빠른 훈련 가능

### Overfitting

![image.png](Perceptron%20&%20DNN%2022327edcc17f803bb43cc5bdf11faddd/image%208.png)

- Underfitting
    - 모델은 데이터를 완전히 학습할 능력 부족
- Overfitting
    - 모델이 많은 parameter가 필요하여 복잡하다.
    - 그러나 일반화되지 않는다.

### Regularization

- 복잡한 모델을 억제하기 위해 최적화 문제를 제한하는 기술
    - 경험하지 않은 데이터에 대한 일반화를 개선하기 위해 사용된다.
1. Dropout
    1. 훈련 중 무작위로 일부 노드의 활성을 0으로 설정한다.
        1. 통상 레이어의 50%의 활성을 드랍시킨다.
        2. 네트워크가 어떤 노드에 의존하지 않도록 강제시킨다.
    
    ![Dropout](Perceptron%20&%20DNN%2022327edcc17f803bb43cc5bdf11faddd/image%209.png)
    
    Dropout
    
2. Early Stopping
    1. overfitting 되기 전에 훈련을 종료시킨다.
    
    ![무제.png](Perceptron%20&%20DNN%2022327edcc17f803bb43cc5bdf11faddd/%E1%84%86%E1%85%AE%E1%84%8C%E1%85%A6.png)
    

## XOR Problem & Visualization

### Logic Gate

- 단일 이진 출력을 생성하는 하나 이상의 이진 입력에 대해 수행되는 논리 연산
- AND Gate: 기본 디지털 논리 게이트 중 하나로 2개의 입력이 모두 true인 경우만 true를 출력한다.
    
    
    | A | B | Y |
    | --- | --- | --- |
    | 0 | 0 | 0 |
    | 0 | 1 | 0 |
    | 1 | 0 | 0 |
    | 1 | 1 | 1 |
- OR Gate: 기본 디지털 논리 게이트 중 하나로 2개의 입력 중 하나라도 true인 경우 true를 출력한다.
    
    
    | A | B | Y |
    | --- | --- | --- |
    | 0 | 0 | 0 |
    | 0 | 1 | 1 |
    | 1 | 0 | 1 |
    | 1 | 1 | 1 |
- XOR Gate: 기본 디지털 논리 게이트 중 하나로 2개의 입력 중 true가 홀수인 경우 true를 출력한다.
    
    
    | A | B | Y |
    | --- | --- | --- |
    | 0 | 0 | 0 |
    | 0 | 1 | 1 |
    | 1 | 0 | 1 |
    | 1 | 1 | 0 |

### XOR Problem

- 비선형 문제
    - AND gate와 OR gate의 경우 선형적 접근으로 해결 가능하나,
    - XOR gate는 비선형적 특성으로 인해 선형적 접근으로 해결 불가능
    - 비선형 문제로 단일 퍼셉트론 연구가 중단

![image.png](Perceptron%20&%20DNN%2022327edcc17f803bb43cc5bdf11faddd/image%2010.png)

- 비선형 문제 접근 방안
    - 다층 구조의 MLP 활용
    - 활성화 함수 활용

### Linear Perceptron

- perceptron
    
    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    class Perceptron(nn.Module):
    	def __init__(self, input_dim: int):
    		super().__init__()
    		
    		# parameter
    		self.W = nn.Parameter(torch.randn(input_dim, 1), requires_grad=True)
    		self.b = nn.Parameter(torch.randn(1, 1), requires_grad=True)
    		
    	def forward(self, x: torch.Tensor) -> torch.Tensor:
    		return x @ self.W + self.b
    ```
    
- AND Gate
    
    ```python
    import torch 
    
    # AND 
    x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=torch.float)
    y = torch.tensor([[0.], [0.], [0.], [1.]], dtype=torch.float)
    
    and_model = Perceptron(2)
    
    # hyperparameter
    lr = 0.01
    loss_fn = nn.MSELoss()
    epochs = 1000
    
    # training model
    while True:
    	and_model = Perceptron(2)
    	optimizer = optim.SGD(and_model.parameters(), lr=lr)
    	
    	for epoch in range(1, epochs+1):
    		optimizer.zero_grad()
    		# forward
    		y_hat = and_model(x)
    		loss = loss_fn(y_hat, y)
    		
    		# backward
    		loss.backward()
    		optimizer.step()
    		
    		if epoch % 100 == 0:
    			print(f"Epoch: {epoch}, Loss: {loss.item(): .2f}")
    	
    	if all(torch.where(and_model(x)>0.5, 1., 0.) == y):
    		break
    
    # print output
    print(and_model(x))
    ```
    
    - visualization
        - 모델의 출력 수식 vs 모델의 출력
            - 0.499236404895782**x1 + 0.499229192733765**x2 - 0.249090000987053
            
            ```python
            model output(model(X)): [-0.2491,  0.2501,  0.2501,  0.7494]
            equation: [-0.2491,  0.2501,  0.2501,  0.7494]
            ```
            
        - 모델의 출력 시각화
            
            ![image.png](Perceptron%20&%20DNN%2022327edcc17f803bb43cc5bdf11faddd/image%2011.png)
            
- OR Gate
    
    ```python
    import torch 
    
    # OR 
    x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=torch.float)
    y = torch.tensor([[0.], [1.], [1.], [1.]], dtype=torch.float)
    
    or_model = Perceptron(2)
    
    # hyperparameter
    lr = 0.01
    loss_fn = nn.MSELoss()
    epochs = 1000
    
    # training model
    while True:
    	and_model = Perceptron(2)
    	optimizer = optim.SGD(or_model.parameters(), lr=lr)
    	
    	for epoch in range(1, epochs+1):
    		optimizer.zero_grad()
    		# forward
    		y_hat = or_model(x)
    		loss = loss_fn(y_hat, y)
    		
    		# backward
    		loss.backward()
    		optimizer.step()
    		
    		if epoch % 100 == 0:
    			print(f"Epoch: {epoch}, Loss: {loss.item(): .2f}")
    	
    	if all(torch.where(or_model(x)>0.5, 1., 0.) == y):
    		break
    
    # print output
    print(or_model(x))
    ```
    
    - visualization
        - 모델의 출력 수식 vs 모델의 출력
            - 0.500494122505188**x1 + 0.500532388687134**x2 + 0.249391213059425
        
        ```python
        model output(model(X)): [0.2494, 0.7499, 0.7499, 1.2504]
        equation: [0.2494, 0.7499, 0.7499, 1.2504]
        ```
        

![image.png](Perceptron%20&%20DNN%2022327edcc17f803bb43cc5bdf11faddd/image%2012.png)

- XOR Gate
    
    ```python
    import torch 
    
    # AND 
    x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=torch.float)
    y = torch.tensor([[0.], [1.], [1.], [0.]], dtype=torch.float)
    
    xor_model = Perceptron(2)
    
    # hyperparameter
    lr = 0.01
    loss_fn = nn.MSELoss()
    epochs = 1000
    
    # training model
    while True:
    	and_model = Perceptron(2)
    	optimizer = optim.SGD(xor_model.parameters(), lr=lr)
    	
    	for epoch in range(1, epochs+1):
    		optimizer.zero_grad()
    		# forward
    		y_hat = xor_model(x)
    		loss = loss_fn(y_hat, y)
    		
    		# backward
    		loss.backward()
    		optimizer.step()
    		
    		if epoch % 100 == 0:
    			print(f"Epoch: {epoch}, Loss: {loss.item(): .2f}")
    	
    	if all(torch.where(xor_model(x)>0.5, 1., 0.) == y):
    		break
    
    # print output
    print(xor_model(x))
    ```
    
    - visualization
        - 모델의 출력 수식 vs 모델의 출력
            - 0.00310595682822168**x1 + 0.00300022657029331**x2 + 0.496378600597382
            
            ```python
            model output(model(X)): [0.4964, 0.4994, 0.4995, 0.5025]
            equation: [0.4964, 0.4994, 0.4995, 0.5025]
            ```
            
        
        ![image.png](Perceptron%20&%20DNN%2022327edcc17f803bb43cc5bdf11faddd/image%2013.png)
        

### Multi-Layer Perceptron

- MLP + activation function
    
    ![image.png](Perceptron%20&%20DNN%2022327edcc17f803bb43cc5bdf11faddd/image%2014.png)
    
    - Output Equation
        - $g(z_1)$: 여기서 $g(\cdot)$은 Sigmoid 함수 적용
            - $z_1=\begin{bmatrix}x_1,&x_2 \end{bmatrix}\begin{bmatrix}w_1 \\ w_2 \end{bmatrix} + b_1=w_1x_1+w_2x_2 + b_1$
            - $g(z_1)=\frac{1}{1+e^{-z_1}}=\frac{1}{1+e^{-(w_1x_1+w_2x_2 + b_1)}}$
            
            ![image.png](Perceptron%20&%20DNN%2022327edcc17f803bb43cc5bdf11faddd/image%2015.png)
            
        - $g(z_2)$
            - $z_2=\begin{bmatrix}x_1,&x_2 \end{bmatrix}\begin{bmatrix}w_3 \\ w_4 \end{bmatrix} + b_2 = w_3x_1+w_4x_2 + b_2$
            - $g(z_2)=\frac{1}{1+e^{-z_1}}=\frac{1}{1+e^{-(w_3x_1+w_4x_2 + b_2)}}$
            
            ![image.png](Perceptron%20&%20DNN%2022327edcc17f803bb43cc5bdf11faddd/image%2016.png)
            
        - $g(\hat y)$
            - $\hat y=\begin{bmatrix}g(z_1),&g(z_2) \end{bmatrix}\begin{bmatrix}w_5 \\ w_6 \end{bmatrix} + b_3 = w_5g(z_1)+w_6g(z_2) + b_3 = w_5\frac{1}{1+e^{-(w_1x_1+w_2x_2 + b_1)}}+w_6\frac{1}{1+e^{-(w_3x_1+w_4x_2 + b_2)}} + b_3$
            - $g(\hat y)=\frac{1}{1+e^{-\hat y}}=\frac{1}{1+e^{-(w_5\frac{1}{1+e^{-(w_1x_1+w_2x_2 + b_1)}}+w_6\frac{1}{1+e^{-(w_3x_1+w_4x_2 + b_2)}} + b_3)}}$
            
            ![image.png](Perceptron%20&%20DNN%2022327edcc17f803bb43cc5bdf11faddd/image%2017.png)
            
    
    ```python
    import torch
    import torch.nn as nn
    
    class MyDenseLayer(nn.Module):
    	def __init__(self, input_dim, output_dim, act_fn='sigmoid', bias=True):
    		# parameters
    		self.W = nn.Parameter(torch.randn(input_dim, output_dim), requires_grad=True)
    		if bias:
    			self.b = nn.Parameter(torch.randn(1, output_dim), requires_grad=True)
    		else: 
    			self.b = None
    		
    		# activation function
    		try:
    			self.activation = getattr(torch, act_fn)
    		except:
    			self.activation = None
    	
    	def forward(self, inputs):
    		z = inputs @ self.W + self.b if isinstance(self.b, nn.Parameter) else inputs @ self.W
    		if self.activation != None:
    			return self.activation(z)
    		return z
    ```
    
- AND Gate
    
    ```python
    import torch
    import torch.nn as nn 
    
    # AND 
    x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=torch.float)
    y = torch.tensor([[0.], [0.], [0.], [1.]], dtype=torch.float)
    
    # hyperparameter
    lr = 0.01
    loss_fn = nn.MSELoss()
    epochs = 1000
    
    # training model
    while True:
    	and_model = nn.Sequential(
    		MyDenseLayer(2, 2, act_fn='sigmoid'),
    		MyDenseLayer(2, 1, act_fn='sigmoid'),
    	)
    	optimizer = optim.SGD(and_model.parameters(), lr=lr)
    	
    	for epoch in range(1, epochs+1):
    		optimizer.zero_grad()
    		# forward
    		y_hat = and_model(x)
    		loss = loss_fn(y_hat, y)
    		
    		# backward
    		loss.backward()
    		optimizer.step()
    		
    		if epoch % 100 == 0:
    			print(f"Epoch: {epoch}, Loss: {loss.item(): .2f}")
    	
    	if all(torch.where(and_model(x)>0.5, 1., 0.) == y):
    		break
    
    # print output
    print(and_model(x))
    ```
    
- OR Gate
    
    ```python
    import torch
    import torch.nn as nn 
    
    # AND 
    x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=torch.float)
    y = torch.tensor([[0.], [1.], [1.], [1.]], dtype=torch.float)
    
    # hyperparameter
    lr = 0.01
    loss_fn = nn.MSELoss()
    epochs = 1000
    
    # training model
    while True:
    	or_model = nn.Sequential(
    		MyDenseLayer(2, 2, act_fn='sigmoid'),
    		MyDenseLayer(2, 1, act_fn='sigmoid'),
    	)
    	optimizer = optim.SGD(and_model.parameters(), lr=lr)
    	
    	for epoch in range(1, epochs+1):
    		optimizer.zero_grad()
    		# forward
    		y_hat = or_model(x)
    		loss = loss_fn(y_hat, y)
    		
    		# backward
    		loss.backward()
    		optimizer.step()
    		
    		if epoch % 100 == 0:
    			print(f"Epoch: {epoch}, Loss: {loss.item(): .2f}")
    	
    	if all(torch.where(or_model(x)>0.5, 1., 0.) == y):
    		break
    
    # print output
    print(or_model(x))
    ```
    
- XOR Gate
    
    ```python
    import torch
    import torch.nn as nn 
    
    # AND 
    x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=torch.float)
    y = torch.tensor([[0.], [1.], [1.], [0.]], dtype=torch.float)
    
    # hyperparameter
    lr = 0.01
    loss_fn = nn.MSELoss()
    epochs = 1000
    
    # training model
    while True:
    	xor_model = nn.Sequential(
    		MyDenseLayer(2, 2, act_fn='sigmoid'),
    		MyDenseLayer(2, 1, act_fn='sigmoid'),
    	)
    	optimizer = optim.SGD(and_model.parameters(), lr=lr)
    	
    	for epoch in range(1, epochs+1):
    		optimizer.zero_grad()
    		# forward
    		y_hat = xor_model(x)
    		loss = loss_fn(y_hat, y)
    		
    		# backward
    		loss.backward()
    		optimizer.step()
    		
    		if epoch % 100 == 0:
    			print(f"Epoch: {epoch}, Loss: {loss.item(): .2f}")
    	
    	if all(torch.where(xor_model(x)>0.5, 1., 0.) == y):
    		break
    
    # print output
    print(xor_model(x))
    ```
    
- 출력 방정식
    
    ```python
    from sympy import symbols, exp, Max
    
    def output_equation(model, act_fn='sigmoid'):
    	# input
    	x1, x2 = symbols('x1, x2')
    	
    	# activation function
    	z = symbols('z')
    	act_func = None
    	if act_fn='sigmoid':
    		act_func = 1 / (1 + exp(-z))
    	elif act_fn='relu':
    		act_func = Max(0, z)
    	elif act_fn='tanh':
    		act_func = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
    	
    	input_layer = np.array([[x1, x2]])
    	output_layer = None
    	for idx, (name, value) in enumerate(model.state_dict().items()):
    		if idx % 2 == 0 and idx == 0:
    			# input_layer @ weights
    			output_layer = input_layer @ value.detach().numpy()
    		elif idx % 2 == 0 and idx > 0:
    			# previous layer @ weights
    			output_layer = output_layer @ value.detach().numpy()
    		elif idx % 2 == 1:
    			# add bias
    			output_layer = output_layer + value.detach().numpy()
    			
    			# activation function
    			if act_func != None:
    				output_layer = np.array([act_func.subs({'z': v}) for v in output_layer[0]])
    
    	output_eq = output_layer[0]
    	if isinstance(output_eq, np.ndarray):
    		return output_eq[0]
    	return output_eq
    ```
    
    - AND Gate
        - output equation
            
            1/(21.2516975548703**exp(-2.44089460372925/(6.8387676076729**exp(-1.78852212429047**x1 - 1.53758156299591**x2) + 1) - 2.49775242805481/(22.7338145774458**exp(-2.08193230628967**x1 - 2.39206647872925*x2) + 1)) + 1)
            
        
        ![image.png](Perceptron%20&%20DNN%2022327edcc17f803bb43cc5bdf11faddd/image%2018.png)
        
    - OR Gate
        - output equation
            
            1/(2.64413862903908**exp(-0.548751294612885/(1.75157253038228**exp(-0.477402716875076**x1 - 2.0777223110199**x2) + 1) - 2.16720271110535/(1.89182811591751**exp(-2.75366592407227**x1 - 1.7736884355545*x2) + 1)) + 1)
            
        
        ![image.png](Perceptron%20&%20DNN%2022327edcc17f803bb43cc5bdf11faddd/image%2019.png)
        
    - XOR Gate
        - output equation
            
            1/(0.182363143616948**exp(1.3712340593338/(0.188261997440905**exp(2.84197354316711**x1 - 3.66640996932983**x2) + 1) + 1.11436450481415/(0.359781791347486**exp(-3.25776386260986**x1 + 1.97481918334961*x2) + 1)) + 1)
            
        
        ![image.png](Perceptron%20&%20DNN%2022327edcc17f803bb43cc5bdf11faddd/image%2020.png)
        

## 참고자료

1. MIT: Introduction to Deep Learning([https://introtodeeplearning.com](https://introtodeeplearning.com/))
2. 모두를 위한 Deep Learning 시즌2([https://deeplearningzerotoall.github.io/season2/](https://deeplearningzerotoall.github.io/season2/))