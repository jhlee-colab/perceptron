import torch 
import numpy as np
from sklearn.decomposition import PCA
from sympy import symbols, exp, Max

def filter_wise_values(model, x, y, loss_fn):
    # trained parameter
    theta_star = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # θ = θ* + ꭤ𝛿 + βη
    # random direction: gaussian, N(0, I)
    # filter-wise normalization: (𝛿, η), N(0, I) * (||θ*|| / ||N(0, I)||)
    delta, eta = {}, {}
    for k, v in theta_star.items():
        rd1 = torch.randn_like(v)
        rd2 = torch.randn_like(v)

        # norm
        norm_theta = v.norm()
        delta[k] = rd1 * (norm_theta / (rd1.norm() + 1e-12))
        eta[k] = rd2 * (norm_theta / (rd2.norm() + 1e-12))

    # set grid
    alphas = np.linspace(-1, 1, 100)
    betas = np.linspace(-1, 1, 100)
    A, B = np.meshgrid(alphas, betas)
    L = np.zeros_like(A, dtype=float)

    # 최종 학습된 파라미터에서 랜덤 방향으로 ꭤ𝛿와 βη 만큼 떨어져 있을 때의 손실 값들
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            # θ = θ* + ꭤ𝛿 + βη
            theta = {
                k: (theta_star[k] + alpha*delta[k] + beta*eta[k]) for k in theta_star
            }
            model.load_state_dict(theta, strict=True)
            with torch.no_grad():
                L[i, j] = loss_fn(model(x), y).item()
    return (A, B, L)

def pca_trajectory_values(model, x, y, loss_fn, theta):
    # 𝛿 = θ - θ0 
    theta = np.stack(theta)
    theta0 = theta[0]
    deltas = theta - theta0

    # pca
    pca = PCA(n_components=2)
    proj = pca.fit_transform(deltas)
    variance_ratio = pca.explained_variance_ratio_

    # pca 방향 벡터
    pca1, pca2 = pca.components_
    #a_val = np.linspace(math.ceil(proj[:, 0].min())-1, math.ceil(proj[:, 0].max())+1, 100)
    #b_val = np.linspace(math.ceil(proj[:, 1].min())-1, math.ceil(proj[:, 1].max())+1, 100)
    a_val = np.linspace(-3, 3, 100)
    b_val = np.linspace(-3, 3, 100)
    A, B = np.meshgrid(a_val, b_val)
    L = np.zeros_like(A)

    # parameter information
    names = list(model.state_dict())
    shapes = [model.state_dict()[n].shape for n in names]
    sizes = [np.prod(s) for s in shapes]
    cumidx = np.cumsum([0] + sizes)

    # loss 
    for i, a in enumerate(a_val):
        for j, b in enumerate(b_val):
            theta_vec = theta0 + a*pca1 + b*pca2
            sd = {}
            for k, name in enumerate(names):
                start, end = cumidx[k], cumidx[k+1]
                sd[name] = torch.from_numpy(theta_vec[start:end].reshape(shapes[k]))
            model.load_state_dict(sd)
            with torch.no_grad():
                L[i, j] = loss_fn(model(x),y).item()
    return (A, B, L, proj, variance_ratio)

def model_output_surface(model):
    X1_val = np.linspace(-0.2, 1.2, 100)
    X2_val = np.linspace(-0.2, 1.2, 100)
    X1, X2 = np.meshgrid(X1_val, X2_val)
    Y = np.zeros_like(X1, dtype=float)

    for i, x1 in enumerate(X1_val):
        for j, x2 in enumerate(X2_val):
            Y[i, j] = model(torch.tensor([[x1, x2]], dtype=torch.float)).item()

    return (X1, X2, Y)

def model_output_equation(model, act_fn):
    if act_fn == 'relu':
        return 
    
    # input symbols
    x1, x2 = symbols('x1, x2')
    # activation function
    z = symbols('z')
    if act_fn == 'sigmoid':
        act_fn = 1 / (1 + exp(-z))
    elif act_fn == 'tanh':
        act_fn = (exp(z) - exp(-z))/(exp(z) + exp(-z))
    elif act_fn == 'relu':
        act_fn = Max(0, z)

    input_layer = np.array([[x1, x2]])
    output_layer = None
    for idx, (name, value) in enumerate(model.state_dict().items()):
        if idx % 2 == 0 and idx == 0:
            # input_layer @ weight
            output_layer = input_layer @ value.detach().numpy()
            #output_layer = output_layer.reshape(1, -1)
        elif idx % 2 == 0 and idx > 0:
            # weight
            output_layer = output_layer @ value.detach().numpy()
        elif idx % 2 == 1:
            # bias
            output_layer = output_layer + value.detach().numpy()

            # activation function
            if act_fn != None:
                output_layer = np.array([act_fn.subs({'z': v}) for v in output_layer[0]])
            #output_layer = output_layer.reshape(1, -1)
    
    output_eq = output_layer[0]
    if isinstance(output_eq, np.ndarray):
        return output_eq[0]
    return output_eq