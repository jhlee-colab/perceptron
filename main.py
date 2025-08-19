import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from exceptions import *
from models import build_model
from trainer import train
from data_processings import filter_wise_values, model_output_surface, model_output_equation
from copy import deepcopy
from plots import plot_model_output, subplots_3d_filter_wise_surface, plot_all
import math

def parser():
    parser = argparse.ArgumentParser(description="training logic gate model and visualize it's model.")
    parser.add_argument('--n_trials', type=int, default=1, help='The number of trials')

    # logic gate 선택
    parser.add_argument('--logic_gate', choices=['and', 'or', 'xor'], default='and', help='Choose which logic gate to use: AND Gate or OR Gate or XOR Gate')

    # build model
    parser.add_argument('--model', choices=['perceptron', 'mlp'], default='perceptron', help='Choose which model type to build: a Perceptron or an MLP')
    parser.add_argument('--hidden_layers', nargs='+', help='the list of output_dim of hidden layers, ex) 4 4 2')
    parser.add_argument('-act', '--activation', choices=['relu', 'sigmoid', 'tanh'], default=None, help='Choose which activation function to assign into layers: Sigmoid or ReLU or Tanh')

    # training
    parser.add_argument('--epochs', type=int, default=2000, help='The number of epoch')
    parser.add_argument('--lr', type=float, default=0.01, help='the learning rate')
    parser.add_argument('--optimizer', choices=['SGD', 'Adam', 'RMSprop', 'NAdam', 'AdamW', 'Adagrad'], default='SGD', help='Choose which optimizer function')
    parser.add_argument('--loss_func', choices=['MSELoss', 'BCELoss'], default='MSELoss')

    # plotting options
    parser.add_argument('--plots', choices=['output', 'filter-wise', 'all'], default=None, help='Choose plotting option')
    parser.add_argument('--print_equation', type=bool, default=False, help="It's enable to print the output of the model.")

    args = parser.parse_args()
    return args

def check_args(args):
    # check hidden_layers
    try: 
        if args.model == 'perceptron' and args.hidden_layers is None:
            args.hidden_layers = []
        else:
            if args.hidden_layers == ['']:
                args.hidden_layers = []
            else:
                args.hidden_layers = [int(x) for x in args.hidden_layers]
    except:
        raise ArgumentError(f'hidden layers should be the list of integer, not {args.hidden_layers}')
    
    # check activation
    try:
        args.activation = getattr(torch, args.activation)
    except:
        args.activation = None
    
    # check optimizer
    try:
        args.optimizer = getattr(optim, args.optimizer)
    except:
        args.optimizer = optim.SGD

    # check loss function
    try:
        args.loss_func = getattr(nn, args.loss_func)()
    except:
        args.loss_func = nn.MSELoss() 
    return args

def training_model(args):
    # input and output
    X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=torch.float)
    y = None
    if args.logic_gate == 'and':
        y = torch.tensor([[0.], [0.], [0.], [1.]], dtype=torch.float)
    elif args.logic_gate == 'or':
        y = torch.tensor([[0.], [1.], [1.], [1.]], dtype=torch.float)
    elif args.logic_gate == 'xor':
        y = torch.tensor([[0.], [1.], [1.], [0.]], dtype=torch.float)

    model = build_model(args.model, 2, 1, hidden_layers=args.hidden_layers, act_fn=args.activation)
    
    data_arr = []
    for nth in range(1, args.n_trials+1):
        print(f'{nth}th tiral')
        
        while True:
            try:
                model = train(model, X, y, args.epochs, optimizer=args.optimizer, lr=args.lr, loss_fn=args.loss_func)
            except KeyboardInterrupt:
                break

            # check output
            if all(torch.where(model(X)>0.5, 1., 0.) == y):
                break
            else:
                for layer in model:
                    layer.reset()
    
        if args.print_equation:
            act_fn = None
            try:
                act_fn=args.activation.__name__
            except: pass 
            eq = model_output_equation(model, act_fn=act_fn)
            print(f'{nth}th model: {eq}')
            print(f'model: {model(X).flatten()}, threshold: {torch.where(model(X)>0.5, 1., 0.).flatten()}')
            eq_output = torch.zeros_like(y)
            for i, xt in enumerate(X):
                eq_output[i] = torch.tensor([eq.subs({'x1': xt[0].item(), 'x2': xt[1].item()})], dtype=torch.float)
            print(f'equation: {eq_output.flatten()}, threshold: {torch.where(eq_output>0.5, 1., 0.).flatten()}')
        if args.plots == 'all':
            filter_wise_value = filter_wise_values(deepcopy(model), X, y, args.loss_func)
            output_surface_value = model_output_surface(model)
            data_arr.append([filter_wise_value, output_surface_value])
        elif args.plots == 'filter-wise':
            filter_wise_value = filter_wise_values(deepcopy(model), X, y, args.loss_func)
            data_arr.append(filter_wise_value)
        elif args.plots == 'output':
            output_surface_value = model_output_surface(model)
            data_arr.append(output_surface_value)
    
    ncol = 1
    nrow = math.ceil(len(data_arr) / ncol)
    if args.plots == 'filter-wise':
        subplots_3d_filter_wise_surface(nrow, ncol, data_arr)
    elif args.plots == 'output':
        plot_model_output(nrow, ncol, data_arr, name=args.logic_gate)
    elif args.plots == 'all':
        nrow = len(data_arr)
        plot_all(nrow, 2, data_arr, name=args.logic_gate)        

    

if __name__ == '__main__':
    args = parser()
    args = check_args(args)

    print(args)
    training_model(args)
