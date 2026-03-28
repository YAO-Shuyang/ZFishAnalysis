from zfish.local_path import *
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from zfish.utils import *

code_id = "5007 - Spatial Preference"
loc = os.path.join(figpath, code_id)
os.makedirs(loc, exist_ok=True)

landmark_pos = [
    (0, 1/18), # Starting Point box
    (1/9, 2/9), # Leaves
    (5/18, 7/18), # Stars
    (4/9, 5/9), # Polygons
    (11/18, 13/18), # Triangles
    (14/18, 15/18), # Gravel
    (16/18, 17/18) # Teeth of Predators
]

with open(os.path.join(loc, "temp.pkl"), 'rb') as f:
    rois_tip, qualified_cells, clusters, meanImgs, lenX, lenY, nplanes = pickle.load(f)

from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
rois = cp.deepcopy(rois_tip)

def calc_density(
    rois: np.ndarray,
    stype: str = 'xy',
    nx = 100,
    ny = 100,
    sigma = 5
):
    if stype == 'xy':
        x, y = rois[:, 0], rois[:, 1]
    elif stype == 'xz':
        x, y = rois[:, 0], rois[:, 2]
    elif stype == 'yz':
        x, y = rois[:, 1], rois[:, 2]
    else:
        raise ValueError("stype must be one of 'xy', 'xz', 'yz'")
    
    
    # smooth the heatmap
    
    if stype == 'xy':
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=[nx, ny], range=[[0-0.5, nx-0.5], [0-0.5, ny-0.5]])
        heatmap = gaussian_filter(heatmap, sigma=sigma)
    else:
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=[200, ny], range=[[0-0.5, nx-0.5], [0-0.5, ny-0.5]])
        interp_heatmap = np.zeros((nx, heatmap.shape[1]))
        # interpolate the heatmap to nx, ny
        for i in range(heatmap.shape[1]):
            heatmap[:, i] = gaussian_filter(heatmap[:, i], sigma=sigma)
            
            x_old = np.linspace(0, 1, heatmap.shape[0])
            x_new = np.linspace(0, 1, nx)
            f = interp1d(x_old, heatmap[:, i], axis=0, kind='linear')
            interp_heatmap[:, i] = f(x_new)
        heatmap = interp_heatmap

    return heatmap

def visualize_cluster(
    rois: np.ndarray, 
    meanImg: np.ndarray,
    color: str = ChannelColors[0],
    lenX: int = 512,
    lenY: int = 512,
):
        
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    # 3d scatter plot of ROIs colored by cluster
    rois[:, 0] = rois[:, 0] % lenX + np.random.rand(rois.shape[0]) - 0.5
    rois[:, 1] = rois[:, 1] % lenY + np.random.rand(rois.shape[0]) - 0.5
    scatter = ax.scatter(
        rois[:, 0], 
        rois[:, 1],  
        rois[:, 2],
        c=color,
        alpha=1, 
        s=3,
        edgecolors=None
    )
    
    # project 2d planes onto the bottom
    #meanImg = meanImg[::4, :, :] + meanImg[1::4, :, :] + meanImg[2::4, :, :] + meanImg[3::4, :, :]
    #meanImg = meanImg[:, ::4, :] + meanImg[:, 1::4, :] + meanImg[:, 2::4, :] + meanImg[:, 3::4, :]
    ny, nx, nz = meanImg.shape
    print(ny, nx, nz)
    
    z_plane = nz+5
    x_plane = -5
    y_plane = ny+5
    
    X_xy, Y_xy = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    img_xy_n = np.mean(meanImg, axis=2)
    img_xy_n = np.clip(img_xy_n, 0, np.percentile(img_xy_n, 90))
    Z_xy = np.zeros_like(X_xy).astype(np.float32)+z_plane

    density_xy = calc_density(rois, stype='xy', nx=nx, ny=ny)
    Z_xy[density_xy > 1e-5] = np.nan
    ax.contourf(
        X_xy, Y_xy, density_xy, 
        zdir='z', offset=z_plane, cmap=sns.dark_palette("#79C", as_cmap=True)
    )
    ax.plot_surface(
        X_xy, Y_xy, Z_xy,
        facecolors=sns.color_palette("gray", as_cmap=True)(img_xy_n.T / np.max(img_xy_n)),
    ) 
    
    img_xz_n = np.mean(meanImg, axis=0)
    img_xz_n = np.clip(img_xz_n, 0, np.percentile(img_xz_n, 90))  
    img_xz_interp = np.zeros((nx, 1000))  
    heatmap_xz = calc_density(rois, stype='xz', nx=nx, ny=nz, sigma=1)
    
    heatmap_xz_interp = np.zeros((nx, 1000))
    for i in tqdm(range(img_xz_n.shape[0])):
        x_old = np.linspace(0, 1, img_xz_n.shape[1])
        x_new = np.linspace(0, 1, 1000)
        f = interp1d(x_old, img_xz_n[i, :], axis=0, kind='linear')
        img_xz_interp[i, :] = f(x_new)
        f = interp1d(x_old, heatmap_xz[i, :], axis=0, kind='linear')
        heatmap_xz_interp[i, :] = f(x_new)
        
    X_xz, Z_xz = np.meshgrid(np.arange(nx), np.linspace(0, nz-1, 1000), indexing='ij')
    Y_xz = np.zeros_like(X_xz).astype(np.float32)+y_plane
    Y_xz[heatmap_xz_interp > 5e-3] = np.nan
    heatmap_xz_interp[heatmap_xz_interp <= 5e-3] = np.nan
    ax.contourf(
        X_xz, heatmap_xz_interp, Z_xz,
        zdir='y', offset=y_plane, cmap=sns.dark_palette("#79C", as_cmap=True)
    )
    ax.plot_surface(
        X_xz, Y_xz, Z_xz,
        facecolors=sns.color_palette("gray", as_cmap=True)(img_xz_interp / np.max(img_xz_n)),
    )
    
    
    img_yz_n = np.mean(meanImg, axis=1)
    img_yz_n = np.clip(img_yz_n, 0, np.percentile(img_yz_n, 90))
    img_yz_interp = np.zeros((ny, 1000))    
    heatmap_yz = calc_density(rois, stype='yz', nx=ny, ny=nz, sigma=1)
    heatmap_yz_interp = np.zeros((ny, 1000))
    for i in tqdm(range(img_yz_n.shape[0])):
        x_old = np.linspace(0, 1, img_yz_n.shape[1])
        x_new = np.linspace(0, 1, 1000)
        f = interp1d(x_old, img_yz_n[i, :], axis=0, kind='linear')
        img_yz_interp[i, :] = f(x_new)
        f = interp1d(x_old, heatmap_yz[i, :], axis=0, kind='linear')
        heatmap_yz_interp[i, :] = f(x_new)
        
    Y_yz, Z_yz = np.meshgrid(np.arange(ny), np.linspace(0, nz-1, 1000), indexing='ij')
    X_yz = np.zeros_like(Y_yz).astype(np.float32)+x_plane
    X_yz[heatmap_yz_interp > 5e-3] = np.nan
    heatmap_yz_interp[heatmap_yz_interp <= 5e-3] = np.nan
    ax.contourf(
        heatmap_yz_interp, Y_yz, Z_yz,
        zdir='x', offset=x_plane, cmap=sns.dark_palette("#79C", as_cmap=True)
    )
    ax.plot_surface(
        X_yz, Y_yz, Z_yz,
        facecolors=sns.color_palette("gray", as_cmap=True)(img_yz_interp / np.max(img_yz_n)),
    )
    
    
    
    ax.set_xlim(0, lenX)
    ax.set_ylim(-5, lenY)
    ax.set_zlim(0, z_plane)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.invert_zaxis()
    plt.show()

meanImg = []
px, py = int(meanImgs.shape[1] / lenX), int(meanImgs.shape[0] / lenY)
for i in range(nplanes):
    nr, nc = i // px, i % px
    meanImg.append(meanImgs[nr*lenY:(nr+1)*lenY, nc*lenX:(nc+1)*lenX])
meanImg = np.stack(meanImg, axis=2)

visualize_cluster(rois[qualified_cells][clusters == 1], meanImg, lenX=lenX, lenY=lenY)