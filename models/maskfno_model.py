import os, sys
module_path = os.path.abspath(os.path.join('/ccc/work/cont002/dam/lehmannf/depot/'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils_ffno import FeedForward, WNLinear

import random

class FactorizedSpectralConv3d(nn.Module):
    def __init__(self, in_dim, out_dim, D1, D2, D3, modes_x, modes_y, modes_z, forecast_ff, backcast_ff,
                 fourier_weight, factor, ff_weight_norm,
                 n_ff_layers, layer_norm, use_fork, dropout):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z
        self.D1 = D1
        self.D2 = D2
        self.D3 = D3
        self.use_fork = use_fork

        self.fourier_weight = fourier_weight
        
        if not self.fourier_weight:
            self.fourier_weight = nn.ParameterList([])
            for n_modes in [self.modes_x, self.modes_y, self.modes_z]:
                weight = torch.FloatTensor(in_dim, out_dim, n_modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)

        if use_fork: # use_fork = False by default
            self.forecast_ff = forecast_ff
            if not self.forecast_ff:
                self.forecast_ff = FeedForward(
                    out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

        self.backcast_ff = backcast_ff
        if not self.backcast_ff: # by default, backcast is defined this way
            self.backcast_ff = FeedForward(
                out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

    def forward(self, x):
        x = self.forward_fourier(x)

        x = x.permute(0, 2, 3, 4, 1)
        b = self.backcast_ff(x)
        b = b.permute(0, 4, 1, 2, 3)
        if self.use_fork:
            f = self.forecast_ff(x)
            f = f.permute(0, 4, 1, 2, 3)
        else:
            f = None
        return b, f

    def forward_fourier(self, x):
        B, I, S1, S2, S3 = x.shape

        # Dimension Z
        x_ftz = torch.fft.rfft(x, dim=-1, norm='ortho')
        out_ft = x_ftz.new_zeros(B, self.out_dim, self.D1, self.D2, self.D3 // 2 + 1)
        
        out_ft[:, :, :min(S1,self.D1), :min(S2,self.D2), :self.modes_z] = torch.einsum(
            "bixyz,ioz->boxyz",
            x_ftz[:, :, :, :, :self.modes_z],
            torch.view_as_complex(self.fourier_weight[2]))[:, :, :min(S1,self.D1), :min(S2,self.D2), :]

        xz = torch.fft.irfft(out_ft, n=self.D3, dim=-1, norm='ortho')

        # Dimension Y
        x_fty = torch.fft.rfft(x, dim=-2, norm='ortho')
        out_ft = x_fty.new_zeros(B, self.out_dim, self.D1, self.D2 // 2 + 1, self.D3)
        
        out_ft[:, :, :min(S1,self.D1), :self.modes_y, :min(S3,self.D3)] = torch.einsum(
            "bixyz,ioy->boxyz",
            x_fty[:, :, :, :self.modes_y, :],
            torch.view_as_complex(self.fourier_weight[1]))[:, :, :min(S1,self.D1), :, :min(S3,self.D3)]

        xy = torch.fft.irfft(out_ft, n=self.D2, dim=-2, norm='ortho')
        
        # Dimension X
        x_ftx = torch.fft.rfft(x, dim=-3, norm='ortho')
        out_ft = x_ftx.new_zeros(B, self.out_dim, self.D1 // 2 + 1, self.D2, self.D3)
        
        out_ft[:, :, :self.modes_x, :min(S2,self.D2), :min(S3,self.D3)] = torch.einsum(
            "bixyz,iox->boxyz",
            x_ftx[:, :, :self.modes_x, :, :],
            torch.view_as_complex(self.fourier_weight[0]))[:, :, :, :min(S2,self.D2), :min(S3,self.D3)]

        xx = torch.fft.irfft(out_ft, n=self.D1, dim=-3, norm='ortho')
        
        # Combining Dimensions
        x = xx + xy + xz

        return x


class FactorizedSpectralConv3d_SharedXY(nn.Module):
    def __init__(self, in_dim, out_dim, D1, D2, D3, modes_x, modes_y, modes_z, forecast_ff, backcast_ff,
                 fourier_weight, factor, ff_weight_norm,
                 n_ff_layers, layer_norm, use_fork, dropout):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        assert modes_x == modes_y
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z
        self.D1 = D1
        self.D2 = D2
        self.D3 = D3
        self.use_fork = use_fork

        self.fourier_weight = fourier_weight
        
        if not self.fourier_weight:
            self.fourier_weight = nn.ParameterList([])
            for n_modes in [self.modes_x, self.modes_z]:
                weight = torch.FloatTensor(in_dim, out_dim, n_modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)

        if use_fork:
            self.forecast_ff = forecast_ff
            if not self.forecast_ff:
                self.forecast_ff = FeedForward(
                    out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

        self.backcast_ff = backcast_ff
        if not self.backcast_ff:
            self.backcast_ff = FeedForward(
                out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

    def forward(self, x):
        x = self.forward_fourier(x)

        x = x.permute(0, 2, 3, 4, 1)
        b = self.backcast_ff(x)
        b = b.permute(0, 4, 1, 2, 3)
        if self.use_fork:
            f = self.forecast_ff(x)
            f = f.permute(0, 4, 1, 2, 3)
        else:
            f = None
        return b, f

    def forward_fourier(self, x):
        B, I, S1, S2, S3 = x.shape

        # Dimension Z
        x_ftz = torch.fft.rfft(x, dim=-1, norm='ortho')
        out_ft = x_ftz.new_zeros(B, self.out_dim, self.D1, self.D2, self.D3 // 2 + 1)
        
        out_ft[:, :, :min(S1,self.D1), :min(S2,self.D2), :self.modes_z] = torch.einsum(
            "bixyz,ioz->boxyz",
            x_ftz[:, :, :, :, :self.modes_z],
            torch.view_as_complex(self.fourier_weight[1]))[:, :, :min(S1,self.D1), :min(S2,self.D2), :]

        xz = torch.fft.irfft(out_ft, n=self.D3, dim=-1, norm='ortho')

        # Dimension Y
        x_fty = torch.fft.rfft(x, dim=-2, norm='ortho')
        out_ft = x_fty.new_zeros(B, self.out_dim, self.D1, self.D2 // 2 + 1, self.D3)
        
        out_ft[:, :, :min(S1,self.D1), :self.modes_y, :min(S3,self.D3)] = torch.einsum(
            "bixyz,ioy->boxyz",
            x_fty[:, :, :, :self.modes_y, :],
            torch.view_as_complex(self.fourier_weight[0]))[:, :, :min(S1,self.D1), :, :min(S3,self.D3)]

        xy = torch.fft.irfft(out_ft, n=self.D2, dim=-2, norm='ortho')
        
        # Dimension X
        x_ftx = torch.fft.rfft(x, dim=-3, norm='ortho')
        out_ft = x_ftx.new_zeros(B, self.out_dim, self.D1 // 2 + 1, self.D2, self.D3)
        
        out_ft[:, :, :self.modes_x, :min(S2,self.D2), :min(S3,self.D3)] = torch.einsum(
            "bixyz,iox->boxyz",
            x_ftx[:, :, :self.modes_x, :, :],
            torch.view_as_complex(self.fourier_weight[0]))[:, :, :, :min(S2,self.D2), :min(S3,self.D3)]

        xx = torch.fft.irfft(out_ft, n=self.D1, dim=-3, norm='ortho')
        
        # Combining Dimensions
        x = xx + xy + xz

        return x

    
    
class ModifyDimensions3d(nn.Module):
    ''' Modify the shape of input from batch, in_width, S1, S2, S3, in_width to batch, out_width, dim1, dim2, dim3 '''
    def __init__(self, dim1, dim2, dim3, in_width, out_width):
        super(ModifyDimensions3d,self).__init__()
        self.conv = nn.Conv3d(in_width, out_width, 1)
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)
        self.dim3 = int(dim3)

    def forward(self, x):
        x = self.conv(x) # modify the number of channels
        
        ft = torch.fft.rfftn(x, dim=[-3,-2,-1], norm='forward')
        d1 = min(ft.shape[2]//2, self.dim1//2)
        d2 = min(ft.shape[3]//2, self.dim2//2)
        d3 = max(min(ft.shape[4]//2, self.dim3//2), 1)
        ft_u = torch.zeros((ft.shape[0], ft.shape[1], self.dim1, self.dim2, self.dim3//2+1), dtype=torch.cfloat, device=ft.device)
        ft_u[:, :, :d1, :d2, :d3] = ft[:, :, :d1, :d2, :d3]
        ft_u[:, :, -d1:, :d2, :d3] = ft[:, :, -d1:, :d2, :d3]
        ft_u[:, :, :d1, -d2:, :d3] = ft[:, :, :d1, -d2:, :d3]
        ft_u[:, :, -d1:, -d2:, :d3] = ft[:, :, -d1:, -d2:, :d3]
        x_out = torch.fft.irfftn(ft_u, s=(self.dim1, self.dim2, self.dim3), norm='forward')

        return x_out

class ShallowTemporalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_time_steps=4096):
        super(ShallowTemporalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
 
        # Create a fixed sinusoidal positional encoding matrix
        pe = torch.zeros(max_time_steps, embedding_dim)
        position = torch.arange(0, max_time_steps, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
 
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
 
        self.register_buffer('pe', pe)
 
    def forward(self, time_indices):
        """
        time_indices: Tensor of shape (batch_size,) or (batch_size, seq_len)
        Returns: Tensor of shape (batch_size, embedding_dim) or (batch_size, seq_len, embedding_dim)
        """
        return self.pe[time_indices]

class maskMIFNO_3D(nn.Module):
    def __init__(self, list_D1, list_D2, list_D3, list_M1, list_M2, list_M3, width, input_dim, output_dim, source_dim=3,
                 branching_index=4, n_layers=4, factor=4, ff_weight_norm=True, n_ff_layers=2, layer_norm=False, padding=8,
                 sharedXY = False, dropout=0.0, time_emb_dim: int = 16):
        super().__init__()
        self.padding = padding # pad the domain if input is non-periodic
        self.width = width
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.source_dim = source_dim # number of features representing the source information
        self.fourier_weight = None
        self.n_layers = n_layers
        self.list_D1 = np.array(list_D1) + padding
        self.list_D2 = np.array(list_D2) + padding
        self.list_D3 = np.array(list_D3) # do not pad the vertical dimension
        self.sharedXY = sharedXY
        self.dropout = dropout
        self.time_emb_dim = time_emb_dim
        self.time_encoding = ShallowTemporalEncoding(self.time_emb_dim, max_time_steps=4096)


        self.branching_index = branching_index
        
        self.P = WNLinear(input_dim, self.width, wnorm=ff_weight_norm)
        
        self.spectral_layers = nn.ModuleList([])
        self.modif_layers = nn.ModuleList([])
        for i in range(self.branching_index):
            if self.sharedXY:
                self.spectral_layers.append(FactorizedSpectralConv3d_SharedXY(in_dim=width, out_dim=width, 
                                                                              D1=self.list_D1[i], D2=self.list_D2[i], D3=self.list_D3[i],
                                                                              modes_x=list_M1[i], modes_y=list_M2[i], modes_z=list_M3[i],
                                                                              forecast_ff=None, backcast_ff=None, fourier_weight=None,
                                                                              factor=factor, 
                                                                              ff_weight_norm=ff_weight_norm,
                                                                              n_ff_layers=n_ff_layers, layer_norm=layer_norm,
                                                                              use_fork=False, dropout=self.dropout))
            else:
                self.spectral_layers.append(FactorizedSpectralConv3d(in_dim=width, out_dim=width, 
                                                                     D1=self.list_D1[i], D2=self.list_D2[i], D3=self.list_D3[i],
                                                                     modes_x=list_M1[i], modes_y=list_M2[i], modes_z=list_M3[i],
                                                                     forecast_ff=None, backcast_ff=None, fourier_weight=None,
                                                                     factor=factor, 
                                                                     ff_weight_norm=ff_weight_norm,
                                                                     n_ff_layers=n_ff_layers, layer_norm=layer_norm,
                                                                     use_fork=False, dropout=self.dropout))
            self.modif_layers.append(ModifyDimensions3d(self.list_D1[i], self.list_D2[i], self.list_D3[i], width, width))

        for i in range(self.branching_index, self.n_layers):
            if self.sharedXY:
                self.spectral_layers.append(FactorizedSpectralConv3d_SharedXY(in_dim=3*width, out_dim=3*width, 
                                                                              D1=self.list_D1[i], D2=self.list_D2[i], D3=self.list_D3[i],
                                                                              modes_x=list_M1[i], modes_y=list_M2[i], modes_z=list_M3[i],
                                                                              forecast_ff=None, backcast_ff=None, fourier_weight=None,
                                                                              factor=factor, 
                                                                              ff_weight_norm=ff_weight_norm,
                                                                              n_ff_layers=n_ff_layers, layer_norm=layer_norm,
                                                                              use_fork=False, dropout=self.dropout))
            else:
                self.spectral_layers.append(FactorizedSpectralConv3d(in_dim=3*width, out_dim=3*width, 
                                                                     D1=self.list_D1[i], D2=self.list_D2[i], D3=self.list_D3[i],
                                                                     modes_x=list_M1[i], modes_y=list_M2[i], modes_z=list_M3[i],
                                                                     forecast_ff=None, backcast_ff=None, fourier_weight=None,
                                                                     factor=factor, 
                                                                     ff_weight_norm=ff_weight_norm,
                                                                     n_ff_layers=n_ff_layers, layer_norm=layer_norm,
                                                                     use_fork=False, dropout=self.dropout))
            self.modif_layers.append(ModifyDimensions3d(self.list_D1[i], self.list_D2[i], self.list_D3[i], 3*width, 3*width))


        ### SOURCE BRANCH
        # size of the variables to concatenate with the part coming from the geology
        self.D1s = list_D1[self.branching_index-1]
        self.D2s = list_D2[self.branching_index-1]
        self.D3s = list_D3[self.branching_index-1] # variable created from the source will be width x D1s x D2s x D3s

        # weights cannot depend on the dimension of the geology, otherwise the FNO is no longer resolution independent
        self.T1 = 2*list_M1[self.branching_index-1]
        self.T2 = 2*list_M2[self.branching_index-1]
        self.T3 = 2*list_M3[self.branching_index-1]
        
        # goes from 3 coordinates to x*y
        self.MLP1 = nn.Sequential(
            nn.Linear(self.source_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.T1*self.T2))
        
        # create the z dimension
        self.Conv2 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(8, self.T3, kernel_size=3, padding='same')
        )
        
        # create the features
        self.Conv3 = nn.Sequential(
            nn.Conv3d(1, self.width//2, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv3d(self.width//2, self.width, kernel_size=3, padding='same')
        )

        self.modif_dim_source = ModifyDimensions3d(self.D1s, self.D2s, self.D3s, self.width, self.width)
        


        ### END PROJECTIONS
        self.QE = nn.Sequential(
            WNLinear(3*self.width + self.time_emb_dim, 128, wnorm=ff_weight_norm),
            WNLinear(128, output_dim, wnorm=ff_weight_norm))
        
        self.QN = nn.Sequential(
            WNLinear(3*self.width + self.time_emb_dim, 128, wnorm=ff_weight_norm),
            WNLinear(128, output_dim, wnorm=ff_weight_norm))
        
        self.QZ = nn.Sequential(
            WNLinear(3*self.width + self.time_emb_dim, 128, wnorm=ff_weight_norm),
            WNLinear(128, output_dim, wnorm=ff_weight_norm))
        

    def forward(self, x, s, grid_bounds):
        ''' x: geology, s: source '''
        grid_bounds_P = grid_bounds.clone()
        z_bound = int(2 * self.transform_position[2])
        z_vals = list(range(0, z_bound + 1, 300))
        zmin_grid, zmax_grid = sorted(random.sample(z_vals, 2))

        grid_bounds_P[:, 2] = zmin_grid # z_min
        grid_bounds_P[:, 5] = zmax_grid # z_max
        grid = self.get_grid(x.shape, x.device, grid_bounds_P)
        #print(fanny)
        x = torch.cat((x, grid), dim=-1)
        x = self.P(x)

        x = x.permute(0, 4, 1, 2, 3)
        if self.padding != 0:
            x = F.pad(x, [0, 0, 0, self.padding, 0, self.padding]) # pad only x and y
            
        for i in range(self.branching_index):
            layer_s = self.spectral_layers[i]
            layer_m = self.modif_layers[i]
            b, _ = layer_s(x)
            x = layer_m(x) + b

        s1 = self.MLP1(s) # s1: _ x 1 x 1024
        s1 = s1.reshape(s1.shape[0], 1, self.T1, self.T2) # _ x 1 x 32 x 32
        
        s2 = self.Conv2(s1) # _ x 32 x 32 x 32
        s2 = torch.unsqueeze(s2, 1) # _ x 1 x 32 x 32 x 32
        
        s3 = self.Conv3(s2) # _ x 16 x 32 x 32 x 32
        s3 = self.modif_dim_source(s3)
        
        y = torch.cat((x + s3, x - s3, x*s3), dim=1)
    
        for i in range(self.branching_index, self.n_layers):
            layer_s = self.spectral_layers[i]
            layer_m = self.modif_layers[i]
            b, _ = layer_s(y)
            y = layer_m(y) + b

        if self.padding != 0:
            b = b[..., :, :-self.padding, :-self.padding, :]
        
        yf = b
        yf = yf.permute(0, 2, 3, 4, 1)

        B, X, Y, T, _ = yf.shape
        time_idx = torch.arange(T, device=yf.device).unsqueeze(0).expand(B, -1)  # (B, T)
        t_enc = self.time_encoding(time_idx)  # (B, T, E)
        t_enc = t_enc.unsqueeze(1).unsqueeze(1).expand(B, X, Y, T, self.time_emb_dim)  # (B, X, Y, T, E)

        yf = torch.cat((yf, t_enc), dim=-1)
        uE = self.QE(yf)
        uN = self.QN(yf)
        uZ = self.QZ(yf)

        return uE, uN, uZ

    
    def get_grid(self, shape, device, grid_bounds):
        # Assuming grid_bounds might be used later, keeping it for now
        #xmin_grid, ymin_grid, xmax_grid, ymax_grid = grid_bounds
        xmin_grid = 0
        ymin_grid = 0
        xmax_grid = 9600
        ymax_grid = 9600
        batchsize, size1, size2, size3 = shape[0], shape[1], shape[2], shape[3]

        # Create grid coordinates directly on the target device
        gridx = torch.linspace(xmin_grid, xmax_grid, size1, device=device, dtype=torch.float32)
        gridy = torch.linspace(ymin_grid, ymax_grid, size2, device=device, dtype=torch.float32)
        gridz = torch.linspace(0, 1, size3, device=device, dtype=torch.float32)

        # Reshape coordinates to be broadcastable
        gridx = gridx.view(1, size1, 1, 1, 1)
        gridy = gridy.view(1, 1, size2, 1, 1)
        gridz = gridz.view(1, 1, 1, size3, 1)

        # Expand coordinates to match the batch size and other dimensions
        # Using expand is more memory efficient than repeat
        gridx = gridx.expand(batchsize, -1, size2, size3, 1)
        gridy = gridy.expand(batchsize, size1, -1, size3, 1)
        gridz = gridz.expand(batchsize, size1, size2, -1, 1)

        # Concatenate along the feature dimension
        grid = torch.cat((gridx, gridy, gridz), dim=-1)
        return grid
    '''
    def get_grid(self, shape, device, grid_bounds):
        batchsize, size1, size2, size3 = shape[0], shape[1], shape[2], shape[3]
        
        # Create lists to hold the grid for each sample in the batch
        gridx_list, gridy_list, gridz_list = [], [], []

        # Iterate over each sample in the batch
        for i in range(batchsize):
            # Unpack the bounds for the current sample
            xmin_grid, ymin_grid, xmax_grid, ymax_grid = grid_bounds[i]

            # Create the grid for the current sample
            _gridx = torch.linspace(xmin_grid, xmax_grid, size1, device=device, dtype=torch.float32)
            _gridy = torch.linspace(ymin_grid, ymax_grid, size2, device=device, dtype=torch.float32)
            _gridz = torch.linspace(0, 1, size3, device=device, dtype=torch.float32)
            
            # Reshape and append to the lists
            gridx_list.append(_gridx.reshape(1, size1, 1, 1, 1).repeat([1, 1, size2, size3, 1]))
            gridy_list.append(_gridy.reshape(1, 1, size2, 1, 1).repeat([1, size1, 1, size3, 1]))
            gridz_list.append(_gridz.reshape(1, 1, 1, size3, 1).repeat([1, size1, size2, 1, 1]))

        # Stack the individual grids into a single batch tensor
        gridx = torch.cat(gridx_list, dim=0)
        gridy = torch.cat(gridy_list, dim=0)
        gridz = torch.cat(gridz_list, dim=0)

        grid=torch.cat((gridx, gridy, gridz), dim=-1)
        return grid
    '''