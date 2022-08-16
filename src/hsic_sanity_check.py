import torch
from loss_capacity.functions import HSIC
device= 'cuda'
x = torch.randn(200,5).to(device)
y = torch.randn(200,50).to(device)

print('dimension unbased scaling')
print('xy HSIC = ', HSIC(x,y,s_x = 1000, s_y=1000))
print('xy HSIC = ', HSIC(x,y,s_x = 100, s_y=100))
print('xy HSIC = ', HSIC(x,y,s_x = 10, s_y=10))
# xy HSIC =  tensor(2.9368e-05, device='cuda:0')

print('dimension based scaling')
print('xy HSIC = ', HSIC(x,y,s_x = 100, s_y=1000))
print('xy HSIC = ', HSIC(x,y,s_x = 10, s_y=100))
print('xy HSIC = ', HSIC(x,y,s_x = 1, s_y=10))

z = x 
#check dep differnt dimension scaling sigma
print('dep')
print('xx HSIC = ', HSIC(x,z,s_x = 1000, s_y=1000))
print('xx HSIC = ', HSIC(x,z,s_x = 10, s_y=10))
print('xx HSIC = ', HSIC(x,z,s_x = 1, s_y=1))
print('xx HSIC = ', HSIC(x,z,s_x = 0.1, s_y=0.1))
# xx HSIC =  tensor(0.0018, device='cuda:0')
# xx HSIC =  tensor(0.0063, device='cuda:0')
# xx HSIC =  tensor(0.0012, device='cuda:0')
# xx HSIC =  tensor(0.0005, device='cuda:0')