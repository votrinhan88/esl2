import pandas as pd
import numpy as np
import torch
    
def get_iris(shuffle:bool=False):
    # Load and process data
    data = pd.read_csv('./data/iris.csv')
    # Map Iris variety to numerical label
    data.loc[data['variety'] == 'Versicolor', 'variety'] = 0
    data.loc[data['variety'] == 'Virginica', 'variety'] = 1
    data.loc[data['variety'] == 'Setosa', 'variety'] = 2
    # Shuffle data, split to train and test set
    X = torch.tensor(data.iloc[:, 0:-1].to_numpy(), dtype=torch.float)
    Y = torch.tensor(data.iloc[:, [-1]].to_numpy(dtype=np.int64))
    if shuffle == True:
        shuffle = torch.randperm(n=X.shape[0])
        X = X[shuffle]
        Y = Y[shuffle]
    return X, Y

def get_clusters_2D(num_clusters, sigma_diag = 0.2, radius = 1, num_examples = 600):
    '''
    num_clusters: number of clusters
    sigma_diag:   
    '''
    pi = torch.acos(torch.zeros(1)).item()*2
    # Mu and Sigma for Gaussian distributions
    mu = torch.cat(
        [radius*torch.cos(torch.arange(num_clusters)*2*pi/num_clusters).unsqueeze(dim = 1),
         radius*torch.sin(torch.arange(num_clusters)*2*pi/num_clusters).unsqueeze(dim = 1)],
        dim = 1)
    sigma = sigma_diag*torch.tensor([[1., 0.], [0., 1.]])
    # Sample from Gaussian distributions
    examples_per_cluster = torch.tensor(num_examples/num_clusters, dtype = torch.int)
    X = torch.cat([torch.distributions.multivariate_normal.MultivariateNormal(mu[cluster, :], sigma).sample([examples_per_cluster]) for cluster in range(num_clusters)], dim = 0)
    y = torch.cat([torch.tensor([[cluster]]*examples_per_cluster) for cluster in range(num_clusters)], dim = 0)
    # Shuffle data
    shuffle_index = torch.randperm(X.size()[0])
    X, y = X[shuffle_index], y[shuffle_index]
    return X, y

if __name__ == '__main__':
    X, y = get_iris()