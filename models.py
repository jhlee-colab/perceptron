from layers import MyDenseLayer
import torch 
import torch.nn as nn


def build_model(model_type, input_dim, output_dim, hidden_layers=[8, 8, 2], act_fn=None):
    model = nn.Sequential()
    layer = MyDenseLayer
    # if model_type == 'mlp':
    #     layer = MyDenseLayer
    # else:
    #     layer = Perceptron

    if model_type == 'perceptron':
        model.append(layer(input_dim, output_dim, act_fn=act_fn))
        return model

    elif model_type == 'mlp' and hidden_layers:
        # input layer
        model.append(layer(input_dim, hidden_layers[0], act_fn=act_fn))

        # hidden layer
        for idx, num in enumerate(hidden_layers[1:]):
            model.append(layer(hidden_layers[idx], num, act_fn=act_fn))
        
        # output layer
        model.append(layer(hidden_layers[-1], output_dim, act_fn=torch.sigmoid))
        print(111, model)
        return model
    return 


if __name__ == '__main__':
    model = build_model('mlp', 2, 1, hidden_layers=[8, 4, 2])
    states = model.state_dict()
    print([8, 4, 2][1:])
    for name in states:
        print(name, states[name].shape)

