from __future__ import division
import torch


import numpy as np
from scipy.stats import gamma



"""
Inputs:
X         n by dim_x matrix
Y         n by dim_y matrix
alph         level of test
Outputs:
testStat    test statistics
thresh        test threshold for level alpha test
"""

def rbf_dot(pattern1, pattern2, deg):
    size1 = pattern1.shape
    size2 = pattern2.shape

    G = np.sum(pattern1*pattern1, 1).reshape(size1[0],1)
    H = np.sum(pattern2*pattern2, 1).reshape(size2[0],1)

    Q = np.tile(G, (1, size2[0]))
    R = np.tile(H.T, (size1[0], 1))

    H = Q + R - 2* np.dot(pattern1, pattern2.T)

    H = np.exp(-H/2/(deg**2))

    return H


def hsic_gam(X, Y, s_x, s_y, alph = 0.1):
    """
    X, Y are numpy vectors with row - sample, col - dim
    alph is the significance level
    auto choose median to be the kernel width
    """
    n = X.shape[0]

    # # ----- width of X -----
    # Xmed = X

    # G = np.sum(Xmed*Xmed, 1).reshape(n,1)
    # Q = np.tile(G, (1, n) )
    # R = np.tile(G.T, (n, 1) )

    # dists = Q + R - 2* np.dot(Xmed, Xmed.T)
    # dists = dists - np.tril(dists)
    # dists = dists.reshape(n**2, 1)

    # width_x = np.sqrt( 0.5 * np.median(dists[dists>0]) )
    # # ----- -----

    # # ----- width of X -----
    # Ymed = Y

    # G = np.sum(Ymed*Ymed, 1).reshape(n,1)
    # Q = np.tile(G, (1, n) )
    # R = np.tile(G.T, (n, 1) )

    # dists = Q + R - 2* np.dot(Ymed, Ymed.T)
    # dists = dists - np.tril(dists)
    # dists = dists.reshape(n**2, 1)

    # width_y = np.sqrt( 0.5 * np.median(dists[dists>0]) )
    # ----- -----

    width_x = s_x
    width_y = s_y
    bone = np.ones((n, 1), dtype = float)
    H = np.identity(n) - np.ones((n,n), dtype = float) / n

    K = rbf_dot(X, X, width_x)
    L = rbf_dot(Y, Y, width_y)

    Kc = np.dot(np.dot(H, K), H)
    Lc = np.dot(np.dot(H, L), H)

    testStat = np.sum(Kc.T * Lc) / n

    varHSIC = (Kc * Lc )**2

    varHSIC = ( np.sum(varHSIC) - np.trace(varHSIC) ) / n / (n-1)

    varHSIC = varHSIC * 2 * (n-4) * (n-5) / n / (n-1) / (n-2) / (n-3)

    K = K - np.diag(np.diag(K))
    L = L - np.diag(np.diag(L))

    muX = np.dot(np.dot(bone.T, K), bone) / n / (n-1)
    muY = np.dot(np.dot(bone.T, L), bone) / n / (n-1)

    mHSIC = (1 + muX * muY - muX - muY) / n

    al = mHSIC**2 / varHSIC
    bet = varHSIC*n / mHSIC
    
    # print('al,bet', al,bet)

    thresh = gamma.ppf(1-alph, al, scale=bet)[0][0]
    if np.isnan(thresh):
        thresh =0

    return torch.Tensor([testStat, thresh])

    
def pairwise_distances(x):
    #x should be two dimensional
    instances_norm = torch.sum(x**2,-1).reshape((-1,1))
    return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()

def GaussianKernelMatrix(x, sigma=1):
    pairwise_distances_ = pairwise_distances(x)
    return torch.exp(-pairwise_distances_ /sigma)

def HSIC(x, y, s_x=1, s_y=1, no_grad = True):
    if no_grad:
        with torch.no_grad():
            m,_ = x.shape #batch size
            K = GaussianKernelMatrix(x,s_x)
            L = GaussianKernelMatrix(y,s_y)
            H = torch.eye(m) - 1.0/m * torch.ones((m,m))
            H = H.cuda()
            result = torch.trace(torch.mm(L,torch.mm(H,torch.mm(K,H))))/((m-1)**2)
    else:
        m,_ = x.shape #batch size
        K = GaussianKernelMatrix(x,s_x)
        L = GaussianKernelMatrix(y,s_y)
        H = torch.eye(m) - 1.0/m * torch.ones((m,m))
        H = H.cuda()
        result = torch.trace(torch.mm(L,torch.mm(H,torch.mm(K,H))))/((m-1)**2)
    return result


# The version with reparamatrization
def pairwise_distances_(x):
    #x should be 3 dimensional
    batch_size,_,_ = x.shape
    instances_norm = torch.sum(x**2,-1).reshape((batch_size,-1,1))
    return -2*torch.bmm(x,torch.permute(x, (0,2,1))) + instances_norm + torch.permute(instances_norm, (0,2,1))

def GaussianKernelMatrix_(x, sigma=1):
    pairwise_distances = pairwise_distances_(x)
    return torch.exp(-pairwise_distances /sigma)
def HSIC_(x, y, s_x=1, s_y=1, no_grad = True):
    # calculate HSIC over a dataset like: [batch_size, num_reparam, dim_x]
    # calculate over dim1,2
    # output: HSIC [batch_size]
    batch_size,_,_ = x.shape
    if no_grad:
        with torch.no_grad():
            _,m,_ = x.shape #num_reparam
            K = GaussianKernelMatrix_(x,s_x) # [batch_size, num_reparam, num_reparam]
            L = GaussianKernelMatrix_(y,s_y)
            H = torch.eye(m) - 1.0/m * torch.ones((m,m))
            BH = H.unsqueeze(0).repeat(batch_size, 1, 1) # repeat batch_size times H
            BH = BH.cuda()
            A = torch.bmm(L,torch.bmm(BH,torch.bmm(K,BH)))
            result = A.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)/((m-1)**2)
    else:
        _,m,_ = x.shape #num_reparam
        K = GaussianKernelMatrix_(x,s_x) # [batch_size, num_reparam, num_reparam]
        L = GaussianKernelMatrix_(y,s_y)
        H = torch.eye(m) - 1.0/m * torch.ones((m,m))
        BH = H.unsqueeze(0).repeat(batch_size, 1, 1) # repeat batch_size times H
        BH = BH.cuda()
        A = torch.bmm(L,torch.bmm(BH,torch.bmm(K,BH)))
        result = A.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)/((m-1)**2) #[batch_size]
    return torch.mean(result) # mean over samples in a batch