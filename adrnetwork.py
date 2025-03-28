import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def CLP(dim_in, dim_out, shape=[256,256], kernel_size=[3,3]):
    return nn.Sequential(
        nn.Conv2d(dim_in, dim_out,kernel_size=kernel_size, padding=kernel_size[0]//2),
        nn.LayerNorm(shape),
        nn.SiLU(),
        nn.Conv2d(dim_out, dim_out,kernel_size=kernel_size, padding=kernel_size[0]//2)
    )
    
class advectionColorBlock(nn.Module):
    def __init__(self, c,  mesh_size, device='cuda'):
        super(advectionColorBlock, self).__init__()
        self.Adv = color_preserving_advection(mesh_size, device)
        
        self.ConvU1 = nn.Conv2d(c, c, kernel_size=[3,3], padding=1)
        self.LNU1    = nn.LayerNorm(normalized_shape=mesh_size)
        self.ConvU2 = nn.Conv2d(c, c,kernel_size=[3,3], padding=1, bias=False)
        self.ConvU2.weight = nn.Parameter(1e-4*torch.randn(c, c, 3, 3))

        self.TimeEmbedU  = nn.Parameter(1e-4*torch.randn(1,c,mesh_size[0],mesh_size[1]))
        self.TNU         = CLP(c, c,mesh_size)
        self.ConvV1 = nn.Conv2d(c, c, kernel_size=[3,3], padding=1)
        self.LNV1    = nn.LayerNorm(normalized_shape=mesh_size)
        self.ConvV2 = nn.Conv2d(c, c,kernel_size=[3,3], padding=1, bias=False)
        self.ConvV2.weight = nn.Parameter(1e-4*torch.randn(c, c, 3, 3))

        self.TimeEmbedV  = nn.Parameter(1e-4*torch.randn(1,c,mesh_size[0],mesh_size[1]))
        self.TNV         = CLP(c, c,mesh_size)
        
    def forward(self, x, t):
        
        nw, nh = x.shape[2], x.shape[3]
        teU = t.reshape([-1,1,1,1])*self.TimeEmbedU
        teU = self.TNU(teU) 
        teV = t.reshape([-1,1,1,1])*self.TimeEmbedV
        teV = self.TNV(teV) 
        
        U = self.ConvU1(x) + teU
        U = self.LNU1(U)
        U = F.silu(U)
        U = self.ConvU2(U)
        
        V = self.ConvV1(x) + teV
        V = self.LNV1(V)
        V = F.silu(V)
        V = self.ConvV2(V)
        U, V = U/nw, V/nw 

        xr = x.reshape(x.shape[0]*x.shape[1], 1, x.shape[2], x.shape[3])
        Ur = U.reshape(x.shape[0]*x.shape[1], 1, x.shape[2], x.shape[3])
        Vr = V.reshape(x.shape[0]*x.shape[1], 1, x.shape[2], x.shape[3])
        
        # Color Conserving Push Forward Operation: Advects pixels in xr by Ur and Vr along x and y axis respectively
        xr = self.Adv(xr, Ur, Vr)
        
        x  = xr.reshape(x.shape)

        return x


class color_preserving_advection(nn.Module):
    def __init__(self, shape, device='cuda'):
        super(color_preserving_advection, self).__init__()
        
        grid_h, grid_w = shape[0], shape[1]
        y, x = torch.meshgrid(torch.linspace(-1, 1, grid_h), torch.linspace(-1, 1, grid_w))
        self.grid = torch.stack((x, y), dim=-1).unsqueeze(0).unsqueeze(0).to(device)

    def forward(self, T, U, V):
        UV = torch.stack((U, V), dim=-1)
        transformation_grid = self.grid + UV
        Th = F.grid_sample(T, transformation_grid.squeeze(1), align_corners=True)

        return Th

# Advection Augmented Convolutional Neural Network
class ADRnet(nn.Module):
    def __init__(self, in_c, hid_c, out_c, nlayers=16, imsz=[256, 256], device='cuda'):
        super(ADRnet, self).__init__()
        
        self.nlayers = nlayers
        self.Open = CLP(in_c, hid_c, imsz)
        
        self.Adv = nn.ParameterList()
        self.DR  = nn.ParameterList()
        
        for i in range(nlayers):
            # Color conserving advection layer
            Advi = advectionColorBlock(hid_c, imsz, device)
            # Diffusion and Reaction Layer: Double convolution layer with nonlineariaty
            DRi  = CLP(hid_c, hid_c, imsz, kernel_size=[5,5])
            
            self.Adv.append(Advi)
            self.DR.append(DRi)
       
        self.Close = nn.Parameter(torch.randn(out_c, hid_c, 1, 1)*1e-2) 
        self.h     = 1/imsz[0]
                
    def forward(self, x, t):
        # Increase the dimensionality
        z = self.Open(x)
        
        # Each residual network layer learns sequential advection, diffusion and reaction in images
        for i in range(self.nlayers):
            # Advection Layer: Learns the advection of color pixels at higher dimension
            dz = self.Adv[i](z,t)
            
            # Learns the diffusion and reaction of color pixels at higher dimension
            dz = self.DR[i](dz)
            
            # Residual Connection
            z  = z + self.h*dz
        
        # Decrease the dimensionality
        x = F.conv2d(z, self.Close) 
        
        return x
