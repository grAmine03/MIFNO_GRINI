import os, sys
module_path = os.path.abspath(os.path.join('/ccc/work/cont002/dam/lehmannf/depot/'))
if module_path not in sys.path:
    sys.path.append(module_path)

import re
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
import pandas as pd
import random

from dataloaders import Normalization

def vector2cube(u, Sx, Sy, Sz, Lv, ixs, iys, izs):
    A = np.zeros((Sx, Sy, Sy))
    i = np.argmax(np.abs(u))
    if i==0:
        X = np.arange(Lv*np.abs(u[0]))*np.sign(u[0])
        Y = u[1] * np.abs(X/u[0]) # np.abs(u[1]/u[0])*X
        Z = u[2] * np.abs(X/u[0]) # np.abs(u[2]/u[0])*X
    elif i==1:
        Y = np.arange(Lv*np.abs(u[1]))*np.sign(u[1])
        X = u[0] * np.abs(Y/u[1]) # np.abs(u[0]/u[1])*Y
        Z = u[2] * np.abs(Y/u[1]) # np.abs(u[2]/u[1])*Y
    else:
        Z = np.arange(Lv*np.abs(u[2]))*np.sign(u[2])
        X = u[0] * np.abs(Z/u[2]) # np.abs(u[0]/u[2])*Z
        Y = u[1] * np.abs(Z/u[2]) # np.abs(u[1]/u[2])*Z
    A[ixs + X.astype(int), iys+Y.astype(int), izs + Z.astype(int)] = 1
    return A


def norm_constant_distance_Vs(s, a, S_in=32, S_in_z=32, Dx=9600, Dy=9600, Dz=9600):
    # reshape tensors with batch size of 1 to arrays without unnecessary dimensions
    s = s.flatten()
    if a.ndim==4 and a.shape[3] == 1:
        a = a[:, :, :, 0]

    hx = Dx/S_in # x size of a mesh element
    hy = Dy/S_in # y size of a mesh element
    hz = Dz/S_in_z # z size of a mesh element
        
    ix = int(s[0]//hx)
    iy = int(s[1]//hy)
    iz = int(S_in_z-1 + s[2]//hz)
    Vs_source = a[ix, iy, iz]
    R = np.sqrt((1e-3*s[2])**2 + (1e-3*Dx/4)**2) # distance to the source
    norm_cst = 1e-6*Vs_source**2 * R
    
    return 1e-2*norm_cst


class GeologyTracesSourceMaskDataset(Dataset):
    def __init__(self, path_data, mask_filename='./data/mask_Nval5000.csv', dir_data=['inputs3D_S32_Z32_T320_fmax5_train'], T_out=320, S_in=32, S_in_z=32, S_out=32, transform_a='normal', N=None, orientation=None,
                 transform_angle=None, transform_position=None, transform_traces=None):
        ''' 
        path_data: string, path to directory with all data
        mask_filename: strin, path to .csv file with positional mask
        dir_data: list of strings, name of folders in path_data where data are stored
        T_out: int, number of time steps in outputs
        S_in: int, number of grid points along x and y, in inputs
        S_in_z: int, number of grid points along z, in inputs
        S_out: int, number of grid points along x and y, in outputs
        transform_a: string, normalization method for inputs. choice between "normal" and "scalar_normal"
        transform_angle: string, normalization method for angle in inputs. choice between 'unit' or None
        transform_position: array of three floats corresponding to the size of x, y, z axes
        transform_traces: string, normalization method for traces. choice between 'distance_traces' or None
        N: number of elements in total
        orientation: "none", "angle" or "moment" to indicate the orientation of the source. if "none": orientation is not included in the input
        '''
        self.path_data = path_data # folder with all data
        self.dir_data = dir_data # name of folder in path_data where data are stored
        self.T_out = T_out # number of time steps in outputs
        self.S_in = S_in # number of grid points along x and y, in inputs
        self.S_in_z = S_in_z # number of grid points along z, in inputs
        self.S_out = S_out # number of grid points along x and y, in outputs
        self.transform_a = transform_a # normalization type
        self.transform_angle = transform_angle # normalization of the angle source
        self.transform_position = transform_position # max of each axis to normalize the source position
        self.transform_traces = transform_traces # normalization type for traces
        self.orientation = orientation # source orientation

        a_mean = np.load( path_data + dir_data[0] + '/a_mean.npy')
        a_std = np.load(path_data + dir_data[0] + '/a_std.npy')

        if self.transform_a == 'scalar_normal':
            self.a_mean = np.mean(a_mean)[:S_in, :S_in, :S_in_z]
            self.a_std = np.mean(a_std)[:S_in, :S_in, :S_in_z]
        else:
            self.a_mean = a_mean[:S_in, :S_in, :S_in_z]
            self.a_std = a_std[:S_in, :S_in, :S_in_z]
        
        if self.transform_a == 'normal' or self.transform_a == 'scalar_normal':
            self.ANorm = Normalization(1, norm_type='normal', x_mean=self.a_mean, x_std=4*self.a_std) # the first argument has no influence since we are imposing the mean and std

        # list of all files
        self.all_files = []
        for indiv_dir_data in dir_data:
            l = os.listdir(self.path_data + indiv_dir_data)
            l = [item for item in l if item[:6] == 'sample']
            if len(l) == 0:
                raise Exception(f"folder {self.path_data + indiv_dir_data} is empty")
            l = sorted(l, key=lambda s: int(re.search(r'\d+', s).group()))
            self.all_files += [self.path_data+indiv_dir_data+'/'+li for li in l]

        if N is not None:
            self.all_files = self.all_files[:N]

        self.mask = pd.read_csv(mask_filename, index_col=[0])
        
    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        f = h5py.File(self.all_files[idx], 'r')
        a = f['a'][:self.S_in, :self.S_in, :self.S_in_z]
        if self.transform_a is not None:
            a = self.ANorm.forward(a)
        a = np.expand_dims(a, axis=3)
        
        uE = f['uE'][:self.S_out, :self.S_out, :]
        uE = np.expand_dims(uE, axis=3)
        
        uN = f['uN'][:self.S_out, :self.S_out, :]
        uN = np.expand_dims(uN, axis=3)
        
        uZ = f['uZ'][:self.S_out, :self.S_out, :]
        uZ = np.expand_dims(uZ, axis=3)

        # Generate random grid bounds for each epoch
        '''
        xmin_grid = random.uniform(0, 2*self.transform_position[0])
        ymin_grid = random.uniform(0, 2*self.transform_position[1])
        xmax_grid = random.uniform(xmin_grid, 2*self.transform_position[0])  # Ensure xmax > xmin
        ymax_grid = random.uniform(ymin_grid, 2*self.transform_position[1])  # Ensure ymax > ymin
        '''
        
        x_bound = int(2 * self.transform_position[0])
        y_bound = int(2 * self.transform_position[1])
        z_bound = int(2 * self.transform_position[2])

        # Generate all possible multiples of 300 within the bounds
        x_vals = list(range(0, x_bound + 1, 300))
        y_vals = list(range(0, y_bound + 1, 300))
        z_vals = list(range(0, z_bound + 1, 300))

        # Randomly select xmin and xmax such that xmin < xmax
        xmin_grid, xmax_grid = sorted(random.sample(x_vals, 2))

        # Randomly select ymin and ymax such that ymin < ymax
        ymin_grid, ymax_grid = sorted(random.sample(y_vals, 2))
        
        
        
        '''
        xmin = self.mask.loc[idx, 'xmin']
        ymin = self.mask.loc[idx, 'ymin']
        
        '''
        
        xmin_grid = 0
        ymin_grid = 0
        xmax_grid = 9600
        ymax_grid = 9600
        
        # normalized mask for the grid position
        
        xmin_grid = xmin_grid/self.transform_position[0]
        ymin_grid = ymin_grid/self.transform_position[1]
        xmax_grid = xmax_grid/self.transform_position[0]
        ymax_grid = ymax_grid/self.transform_position[0]
        

        
        tmin_grid = 0
        tmax_grid = 320
        grid_bounds = np.array([xmin_grid, ymin_grid, tmin_grid, xmax_grid, ymax_grid, tmax_grid], dtype=np.float32)        
        print(f'\n \nin data loader: idx={idx}')
        print(f'in data loader: xmin_grid={xmin_grid:.3f}, ymin_grid={ymin_grid:.3f}, xmax_grid={xmax_grid:.3f}, ymax_grid={ymax_grid:.3f}')
        
        if 's' in f.keys():
            s_raw = f['s'][:]
        else:
            s_raw = np.array([4800, 4800, -8400])

        if self.transform_traces == 'distance_Vs':
            norm_cst = norm_constant_distance_Vs(s_raw, f['a'][:])
            uE *= norm_cst
            uN *= norm_cst
            uZ *= norm_cst
        else:
            norm_cst = 1
        
        # Translate the source by the mask
        s_raw[0] += xmin_grid * self.transform_position[0]
        s_raw[1] += ymin_grid * self.transform_position[1]
            
        if self.transform_position is not None:
            s = s_raw/np.array(self.transform_position)
        else:
            s = s_raw
        s = s.astype(np.float32)
        print(f'in data loader: original source ({f["s"][0]}, {f["s"][1]}, {f["s"][2]}) - normalized source position in mask ({s[0]:.3f}, {s[1]:.3f}, {s[2]:.3f})')
        
        if self.orientation == 'angle':
            if 'angle' in f.keys():
                ang = f['angle'][:].astype(np.float32)
            else:
                ang = np.array([48, 45, 88], dtype=np.float32) # default moment tensor

            if self.transform_angle == 'unit':
                ang[0] = (ang[0] - 180)/180
                ang[1] = (ang[1] - 45)/45
                ang[2] = (ang[2] - 180)/180

            s = np.concatenate([s, ang])
                
        elif self.orientation == 'moment':
            s = np.concatenate([s, f['moment'][:].astype(np.float32)])
        s = np.expand_dims(s, axis=0)

        f.close()
        '''
        print (f'uE shape: {uE.shape}, uN shape: {uN.shape}, uZ shape: {uZ.shape}, a shape: {a.shape}, s shape: {s.shape}')
        
        t0=np.random.randint(0, self.T_out//2)
        tvec=t0+self.T_out//2
        print(f't0: {t0}, tvec: {tvec}')
        return a, uE, uN, uZ, s,(t0,tvec), grid_bounds, norm_cst
        '''
        return a, uE, uN, uZ, s, grid_bounds, norm_cst
