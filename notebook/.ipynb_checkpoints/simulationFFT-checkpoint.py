#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# simulationFFT.py
# Created: April 6th, 2018

import numpy as np
from numpy.fft import fftn, fftshift, ifftn
from numpy.random import uniform as rand


def transform_distribution(grid, new_distribution):
    """ Transforms grid to new distribution."""
    old_distribution = np.sort(grid.flatten())
    new_distribution = np.sort(np.random.choice(
        new_distribution, size=grid.size))
    d = dict(zip(old_distribution, new_distribution))
    return np.vectorize(d.get)(grid)


def simulFFT(nx, ny, nz, mu, sill, m, lx, ly, lz, seed=1):
    '''
    Python implementation of GAIA lab's MGSimulFFT.m available here: 
    http://wp.unil.ch/gaia/downloads/
    Author: Noah Athens
    '''

    """ Performs unconditional simulation with specified mean, variance,
    and correlation length.
    Input:
    nx: number of cells in the  x-direction
    ny: number of cells in the  y-direction
    nz: number of cells in the  z-direction
    mu: Mean of R.V.
    sill: total variance of R.V.
    m: variogram model type ("Exponential" or "Gaussian")
    lx: correlation length x direction (number of cells)
    ly: correlation length y direction (number of cells)
    lz:: correlation length z direction (number of cells)
    seed: numpy random seed
    
    Output:
    unconditional simulation
    """
    np.random.seed(seed)
    if nz == 0:
        nz = 1  # 2D case
    xx, yy, zz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz))
    points = np.stack((xx.ravel(), yy.ravel(), zz.ravel())).T
    centroid = points.mean(axis=0)
    length = np.array([lx, ly, lz])
    h = np.linalg.norm((points - centroid) / length,
                       axis=1).reshape((ny, nx, nz))

    if m == 'Exponential':
        c = np.exp(-3 * h) * sill
    elif m == 'Gaussian':
        c = np.exp(-3 * h**2) * sill
    else:
        raise(Exception('For m enter either "Exponential" or "Gaussian"'))

    grid = fftn(fftshift(c)) / (nx * ny * nz)
    grid = np.abs(grid)
    grid[0, 0, 0] = 0  # reference level
    ran = np.sqrt(grid) * np.exp(1j * np.angle(fftn(rand(size=(ny, nx, nz)))))
    grid = np.real(ifftn(ran * nx * ny * nz))
    std = np.std(grid)
    if nx == 1 or ny == 1 or nz == 1:
        grid = np.squeeze(grid)
    return grid / std * np.sqrt(sill) + mu

def truncate_grf(grid, 
                 proportions=(.5,.5),
                 vals=(0,1),
                 log10trans=True, 
                 plotyn=False, 
                 saveyn=False):
    '''
    Categorizes gridded data into specified proportions by truncating the empirical CDF
    of the input data
    
    Input:
    grid: gridded data (numpy array)
    proportions: iterable of proportions for each category (must sum to 1)
    vals: vals to assign to each category
    log10trans: transform the data out of log10-space (bool)
    plotyn: plot the output grid? (bool)
    saveyn: save the output grid? (bool)
    
    Output:
    outgrid: categorized grid (numpy array)
    '''
    
    
    grid_cutoffs = []
    for q in np.cumsum(proportions):
        grid_cutoffs.append(np.quantile(grid, q))

    if plotyn:
        h = plt.hist(grid.flatten())
        for cutoff in grid_cutoffs:
            plt.vlines(cutoff, 0, 14000)
        plt.show()

    outgrid = np.ones(grid.shape, dtype=np.float32)
    for i, cutoff in reversed(list(enumerate(grid_cutoffs))):
        outgrid[np.where(grid < cutoff)] = vals[i]

    if plotyn:
        f, axs = plt.subplots(2, 1, sharex=True)
        axs[0].imshow(grid[:, 0, :])
        axs[1].imshow(outgrid[:, 0, :])
        if saveyn:
            plt.savefig(m.MC_file.parent.joinpath(
                'Truncated_GRF.png').as_posix(), resolution=300)
    if log10trans:
        return 10**outgrid
    else:
        return outgrid