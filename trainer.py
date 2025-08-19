import torch
import torch.nn as nn
import torch.optim as optim 


def train(model, x, y, epochs=2000, optimizer=optim.SGD, lr=0.01, loss_fn=nn.MSELoss()):
    # optimizer 
    try:
        optimizer = optimizer(model.parameters(), lr=lr)
    except:
        optimizer = optim.SGD(model.parameters(), lr=lr)
    
    for epoch in range(1, epochs+1):
        optimizer.zero_grad()

        y_hat = model(x)
        loss = loss_fn(y_hat, y)

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")
    return model