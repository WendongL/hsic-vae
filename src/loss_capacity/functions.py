import torch

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