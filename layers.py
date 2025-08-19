import torch
import torch.nn as nn


class MyDenseLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bias: bool=True, act_fn=None) -> None:
        super().__init__()
    
        # initialize Parameters
        self.W = nn.Parameter(nn.init.xavier_normal_(torch.empty(input_dim, output_dim)), requires_grad=True)
        if bias:
            self.b = nn.Parameter(nn.init.xavier_normal_(torch.empty(1, output_dim)), requires_grad=True)
        else:
            self.b = None

        self.activation = act_fn
    
    def __str__(self):
        return f'Dense Layer({self.W.shape[0]}, {self.W.shape[1]})'

    def reset(self):
        self.W.data.copy_(nn.init.xavier_normal_(torch.empty_like(self.W.data)))
        if isinstance(self.b, nn.Parameter):
            self.b.data.copy_(nn.init.xavier_normal_(torch.empty_like(self.b.data)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # forward propagation
        z = inputs @ self.W + self.b if isinstance(self.b, nn.Parameter) else inputs @ self.W
        if self.activation is not None:
            return self.activation(z)
        return z


# class Perceptron(nn.Module):
#     def __init__(self, input_dim: int, act_fn=None):
#         super().__init__()
		
# 		# parameter
#         self.W = nn.Parameter(torch.randn(input_dim, 1), requires_grad=True)
#         self.b = nn.Parameter(torch.randn(1, 1), requires_grad=True)
		
#     def reset(self):
#         self.W.data.copy_(torch.randn_like(self.W.data))
#         self.b.data.copy_(torch.randn_like(self.b.data))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return x @ self.W + self.b


if __name__ == '__main__':
    layer = MyDenseLayer(4, 2, bias=False, act_fn=torch.relu)

    x = torch.randn(10, 4)
    y = layer(x)
    print(x.shape, y.shape)
    print(y[:3])

    layer = MyDenseLayer(4, 2)
    y = layer(x)
    print(x.shape, y.shape)
    print(y[:3])