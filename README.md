# Perceptron & MLP
- 퍼셉트론과 멀티레이어 페셉트론에 대한 이해와 시각화에 대한 코드

### Arguments
| argument | description |
| - | - |
| n_trials | The number of trials: default 1 |
| --logic_gate | Choose a logic gate: [**AND**, OR, XOR] |
| --model | Choice a model: [**pertceptron**, mlp] |
| --hidden_layers | hidden layers: ex) 2 4 4 |
| --activation | Choice a activation function: [**Sigmoid**, ReLU, Tanh] |
| --epochs | the number of epoch: default 2,000 |
| --lr | the learning rate: default 0.01 |
| --optimizer | Choice a optimizer: [**SGD**, Adam, RMSprop, NAdam, AdamW, Adagrad] |
| --loss_func | Choice a loss function: [**MSELoss**, BCELoss] |
| --plots | Choose a plot item: [output, filter-wise, all] |
| --print_equation | Print the output equation of the model or not |

### Execution
```shell
python main.py --logic_gate or --model perceptron  --activation sigmoid --print_equation True  --optimizer Adam --lr 0.001 --plots all
```