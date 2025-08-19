from matplotlib import pyplot as plt
import numpy as np


def subplots_3d_filter_wise_surface(nrow, ncol, data_arr):
    if len(data_arr) > 1:
        fig = plt.figure()
        for idx, line in enumerate(data_arr):
            A, B, L = line
            print(A.shape, B.shape, L.shape)
            ax = fig.add_subplot(nrow, ncol, idx+1, projection='3d')
            ax.plot_surface(A, B, L, cmap='coolwarm', edgecolor='none', alpha=1, label='Loss Landscape')
            ax.set_xlabel('α (δ scale)')
            ax.set_ylabel('β (η scale)')
            ax.set_zlabel('Loss L(θ*+αδ+βη)')
            ax.legend(loc='best')
        fig.suptitle('Filter-Normalized Loss Landscape')
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.2, top=0.9, bottom=0.1)
    else:
        A, B, L = data_arr[0]
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot_surface(A, B, L, cmap='coolwarm', edgecolor='none', alpha=1, label='Loss Landscape')
        ax.set_xlabel('α (δ scale)')
        ax.set_ylabel('β (η scale)')
        ax.set_zlabel('Loss L(θ* + αδ + βη)')
        plt.title('Filter-Normalized Loss Landscape')
        ax.legend(loc='best')
    plt.show()

def subplots_2d_trajectory(nrow, ncol, data_arr, step=1):
    fig = plt.figure()
    if len(data_arr) > 1:
        for idx, line in enumerate(data_arr):
            A, B, L, proj, evr = line
            ax = fig.add_subplot(nrow, ncol, idx+1)
            ax.contour(A, B, L, linewidths=0.5, cmap='coolwarm')
            ax.contourf(A, B, L, cmap='coolwarm', alpha=0.1)
            X = np.concatenate([proj[::step, [0]], proj[-1, [0]].reshape(-1,1)]).flatten()
            Y = np.concatenate([proj[::step, [1]], proj[-1, [1]].reshape(-1,1)]).flatten()
            ax.plot(X, Y, 'r-o', ms=3, label='Trajectory')
            ax.scatter(proj[0,0], proj[0,1], c='white', edgecolors='k', s=80, label='Start')
            ax.scatter(proj[-1,0], proj[-1, 1], c='black', s=80, label='End')
            ax.set_xlabel(f'1st PCA Component: {evr[0]*100: .1f}%')
            ax.set_ylabel(f'2nd PCA Component: {evr[1]*100: .1f}%')
            ax.legend(loc='best', ncol=3, fontsize=8)
        fig.suptitle('Loss Contour & PCA Trajectory')
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.5, hspace=0.5, top=0.9, bottom=0.1)
    else:
        A, B, L, proj, evr = data_arr[0]
        ax = fig.add_subplot(111)
        ax.contour(A, B, L, levels=14, linewidths=0.5, cmap='coolwarm')
        ax.contourf(A, B, L, levels=14, cmap='coolwarm', alpha=0.1)
        X = np.concatenate([proj[::step, [0]], proj[-1, [0]].reshape(-1,1)]).flatten()
        Y = np.concatenate([proj[::step, [1]], proj[-1, [1]].reshape(-1,1)]).flatten()
        ax.plot(X, Y, 'r-o', ms=3, label='Trajectory')
        ax.scatter(proj[0,0], proj[0,1], c='white', edgecolors='k', s=80, label='Start')
        ax.scatter(proj[-1,0], proj[-1, 1], c='black', s=80, label='End')
        ax.set_xlabel(f'1st PCA Component: {evr[0]*100: .1f}%')
        ax.set_ylabel(f'2nd PCA Component: {evr[1]*100: .1f}%')
        ax.set_title('Loss Contour & PCA Trajectory')
        ax.legend(loc='best', ncol=3, fontsize=8)
    plt.show()


def subplots_3d_trajectory(nrow, ncol, data_arr, hist_loss, step=1):
    fig = plt.figure()
    if len(data_arr) > 1:
        for idx, line in enumerate(data_arr):
            A, B, L, proj, evr = line
            print(A.shape, B.shape, L.shape)
            ax = fig.add_subplot(nrow, ncol, idx+1, projection='3d')
            #ax.contour(A, B, L, levels=14, linewidths=0.5, cmap='coolwarm')
            ax.contourf(A, B, L, cmap='coolwarm', alpha=0.9)
            X = np.concatenate([proj[::step, [0]], proj[-1, [0]].reshape(-1,1)]).flatten()
            Y = np.concatenate([proj[::step, [1]], proj[-1, [1]].reshape(-1,1)]).flatten()
            Z = hist_loss[::step]+[hist_loss[-1]]
            print('3d', proj.shape)
            ax.plot(X, Y, Z, 'o', ms=3, label='Trajectory')
            ax.scatter(proj[0,0], proj[0,1], Z[0], c='white', edgecolors='k', s=80, label='Start')
            ax.scatter(proj[-1,0], proj[-1, 1], Z[-1], c='black', s=80, label='End')
            ax.set_xlabel(f'1st PCA Component: {evr[0]*100: .1f}%')
            ax.set_ylabel(f'2nd PCA Component: {evr[1]*100: .1f}%')
            ax.legend(loc='best', ncol=3, fontsize=8)
        fig.suptitle('Loss Contour & PCA Trajectory')
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.5, hspace=0.5, top=0.9, bottom=0.1)
    else:
        A, B, L, proj, evr = data_arr[0]
        ax = fig.add_subplot(111, projection='3d')
        #ax.contour(A, B, L, levels=14, linewidths=0.5, cmap='coolwarm')
        ax.contourf(A, B, L, cmap='coolwarm', alpha=0.9)
        X = np.concatenate([proj[::step, [0]], proj[-1, [0]].reshape(-1,1)]).flatten()
        Y = np.concatenate([proj[::step, [1]], proj[-1, [1]].reshape(-1,1)]).flatten()
        Z = hist_loss[::step]+[hist_loss[-1]]
        ax.plot(X, Y, Z, 'o', ms=3, label='Trajectory')
        ax.scatter(proj[0,0], proj[0,1], Z[0], c='white', edgecolors='k', s=80, label='Start')
        ax.scatter(proj[-1,0], proj[-1, 1], Z[-1], c='black', s=80, label='End')
        ax.set_xlabel(f'1st PCA Component: {evr[0]*100: .1f}%')
        ax.set_ylabel(f'2nd PCA Component: {evr[1]*100: .1f}%')
        ax.set_title('Loss Contour & PCA Trajectory')
        ax.legend(loc='best', ncol=3, fontsize=8)
    plt.show()

def plot_model_output(nrow, ncol, data_arr, name=None):
    fig = plt.figure()
    if len(data_arr) > 1:
        for idx, line in enumerate(data_arr):
            X1, X2, Y = line
            ax = fig.add_subplot(nrow, ncol, idx+1, projection='3d')
            ax.plot_surface(X1, X2, Y, cmap='coolwarm', label='Output Landscape')
            ax.set_xlabel(f'X1')
            ax.set_ylabel(f'X2')
            ax.legend(loc='best', ncol=3, fontsize=8)
        fig.suptitle(f'output surface of the {name.upper()} model')
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.5, hspace=0.5, top=0.9, bottom=0.1)
    else:
        X1, X2, Y = data_arr[0]
        ax = fig.add_subplot(nrow, ncol, 1, projection='3d')
        ax.plot_surface(X1, X2, Y, cmap='coolwarm', label='Output Landscape')
        ax.set_xlabel(f'X1')
        ax.set_ylabel(f'X2')
        ax.legend(loc='best', ncol=3, fontsize=8)
        ax.set_title(f'output surface of the {name.upper()} model')
    plt.show()

def plot_all(nrow, ncol, data_arr, name=None):
    if nrow > 1:
        fig = plt.figure()
        for idx, line in enumerate(data_arr):
            (A, B, L), (X1, X2, Y) = line

            ax1 = fig.add_subplot(nrow, 2, idx*2+1, projection='3d')
            ax1.plot_surface(A, B, L, cmap='coolwarm', edgecolor='none', alpha=1, label='Loss Landscape')
            ax1.set_xlabel('α (δ scale)')
            ax1.set_ylabel('β (η scale)')
            ax1.set_zlabel('Loss L(θ* + αδ + βη)')
            ax1.set_title(f'Loss Landscape of the {name.upper()} model')
            ax1.legend(loc='best')

            ax2 = fig.add_subplot(nrow, 2, idx*2+2, projection='3d')
            ax2.plot_surface(X1, X2, Y, cmap='coolwarm', label='Output Landscape')
            ax2.set_xlabel(f'X1')
            ax2.set_ylabel(f'X2')
            ax2.legend(loc='best', ncol=3, fontsize=8)
            ax2.set_title(f'Output Landscape of the {name.upper()} model')
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.2, top=0.9, bottom=0.1)
    else:
        (A, B, L), (X1, X2, Y) = data_arr[0]
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.plot_surface(A, B, L, cmap='coolwarm', edgecolor='none', alpha=1, label='Loss Landscape')
        ax1.set_xlabel('α (δ scale)')
        ax1.set_ylabel('β (η scale)')
        ax1.set_zlabel('Loss L(θ* + αδ + βη)')
        ax1.set_title(f'Loss Landscape of the {name.upper()} model')
        ax1.legend(loc='best')

        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.plot_surface(X1, X2, Y, cmap='coolwarm', label='Output Landscape')
        ax2.set_xlabel(f'X1')
        ax2.set_ylabel(f'X2')
        ax2.legend(loc='best', ncol=3, fontsize=8)
        ax2.set_title(f'Output Landscape of the {name.upper()} model')
    plt.show()