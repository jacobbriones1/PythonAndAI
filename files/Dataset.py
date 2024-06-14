from numpy import random
import numpy as np
import matplotlib.pyplot as plt

def generate_points(r_min, r_max, N=250, theta_min=None, theta_max = None):
      r_dist = random.uniform(r_min, r_max, N)
      if theta_min and theta_max:
        theta_dist = random.uniform(theta_min, theta_max, N)
      else:
        theta_dist = random.uniform(0, 2*np.pi, N)

      return np.array([r_dist*np.cos(theta_dist), r_dist*np.sin(theta_dist)])

def gaussianDataset(r_1=1, r_2=1.7, n1=200, n2=200):
    X0 = generate_points(0, r_1,N=n1)
    
    X1 = generate_points(r_1, r_2, N=n2)
    Y0 = [[1.,0] for i in range(n1)]
    Y1 = [[0., 1] for i in range(n2)]
    
    return [(X0,Y0),(X1,Y1)]
    

def plot_gaussian(X0, X1):
    fig, ax = plt.subplots()

    ax.scatter(X0[0],X0[1],color='blue',edgecolor='k',lw=1)
    ax.scatter(X1[0],X1[1],color='orange',edgecolor='k',lw=1)
    R0=np.max(np.sqrt(X0[0]*X0[0]+X0[1]*X0[1]))
    R1=np.max(np.sqrt(X1[0]*X1[0]+X1[1]*X1[1]))
    ax.plot(R0*np.cos(np.linspace(0,2*np.pi)),R0*np.sin(np.linspace(0,2*np.pi)),'k--',lw=0.5)
    ax.plot(R1*np.cos(np.linspace(0,2*np.pi)),R1*np.sin(np.linspace(0,2*np.pi)),'k--',lw=0.5)
    ax.set_title("Two Class Dataset")
    ax.legend(['0','1'],loc="upper right")
    plt.show()

