#!/usr/bin/python
# -*- coding: utf-8 -*-

import datetime
from functools import partial
import h5py
import itertools
from math import ceil, floor
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as drv
import scipy
from scipy import exp, sinh, cosh, cos, sin, sqrt, pi
from scipy.interpolate import interp1d
import sys
import time
from tomography_tools import build_mesh, calc_h, gamma, setup_reconstructions_group


RADON_KERNEL = """
#include <math_constants.h>

#define ORDER {}
#define NO_ANGLES {}
#define NO_PULSES {}

__constant__ float rsq4pi;
__constant__ float sqeta;
__constant__ float h;
__constant__ float four_pi_gamma;
__constant__ float y;
__constant__ float n2[ORDER];
__constant__ float pre_s1[ORDER];
__constant__ float pre_s2[ORDER];
__constant__ float pre_s3[ORDER];
__constant__ float cos_phi[NO_ANGLES];
__constant__ float sin_phi[NO_ANGLES];

__global__ void K_l(float *quadratures,
                    float *Q, float *P, float *Kb) {{
  extern __shared__ float k[];
  uint quad_idx = blockIdx.y*blockDim.y+threadIdx.y;
  uint phi_idx = (quad_idx/NO_PULSES)%NO_ANGLES;
  uint q_idx = blockIdx.x;
  uint p_idx = blockIdx.x;
  float x = quadratures[quad_idx]/sqeta;
  float z = (Q[q_idx]*cos_phi[phi_idx] + P[p_idx]*sin_phi[phi_idx] - x)/h;
  float zy = z/y;
  float zy2 = powf(zy, 2);
  float s1 = 0.;
  float s2 = 0.;
  float s3 = 0.;
  for(int n=0; n<ORDER; n++) {{
    float denom = n2[n] + zy2;
    s1 += pre_s1[n]/denom;
    s2 += pre_s2[n]/denom;
    s3 += pre_s3[n]/denom;
  }}
  k[threadIdx.y] = zy*__sinf(z)*s3;
  k[threadIdx.y] += zy2*(s1 - __cosf(z)*s2);
  k[threadIdx.y] /= sqrtf(CUDART_PI_F);
  k[threadIdx.y] += __cosf(z)*(expf(powf(y, 2.))-rsq4pi) + (rsq4pi - 1.);
  k[threadIdx.y] /= four_pi_gamma;
  __syncthreads();
  for(unsigned int s=1; s < blockDim.y; s *= 2) {{
    if (threadIdx.y % (2*s) == 0) {{
      k[threadIdx.y] += k[threadIdx.y + s];
    }}
    __syncthreads();
  }}
  if (threadIdx.y==0) Kb[blockIdx.x*gridDim.y+blockIdx.y] = k[0];
}}
"""


REDUCTION_KERNEL = """
__global__ void reduction(float *Kb, float *W) {{
  extern __shared__ float k[];
  k[threadIdx.y] = Kb[blockIdx.x*blockDim.y+threadIdx.y];
  __syncthreads();
  for(unsigned int s=1; s < blockDim.y; s *= 2) {{
    if (threadIdx.y % (2*s) == 0) {{
      k[threadIdx.y] += k[threadIdx.y + s];
    }}
    __syncthreads();
  }}
  if (threadIdx.y==0) W[blockIdx.x] = k[0];
}}
"""


def estimate_position_from_quadratures(eta, angles, quadratures, N_phi=30, N_x=101):
    X = quadratures.reshape(100, 800)/sqrt(eta)
    phi_edges = scipy.linspace(0, 2.*scipy.pi, N_phi)
    phi_centers = (phi_edges[:-1]+phi_edges[1:])/2.
    phi_idx = scipy.digitize(angles, phi_edges)
    xs = [X[phi_idx==n+1] for n in range(len(phi_centers))]
    means = scipy.array([scipy.mean(x) for x in xs])
    stds = scipy.array([scipy.std(x) for x in xs])
    m = interp1d(phi_centers, means)
    return -m(pi), m(pi/2.), stds.max()


class CudaCalculator(object):
    def __init__(self, eta, beta, L, angles, no_pulses, order=5):
        self.angles = angles
        no_angles = angles.shape[0]
        self.mod_K = SourceModule(RADON_KERNEL.format(order, no_angles, no_pulses))
        self.K_gpu = self.mod_K.get_function("K_l")
        self.mod_reduction = SourceModule(REDUCTION_KERNEL)
        self.reduction_gpu = self.mod_reduction.get_function("reduction")
        self.eta = eta
        self.gamma = gamma(eta)
        self.beta = beta
        self.L = L
        self.h = calc_h(L, beta, eta)
        drv.memcpy_htod(self.mod_K.get_global("rsq4pi")[0], scipy.array([1./sqrt(4.*pi)], dtype=scipy.float32))
        drv.memcpy_htod(self.mod_K.get_global("sqeta")[0], scipy.array([sqrt(self.eta)], dtype=scipy.float32))
        drv.memcpy_htod(self.mod_K.get_global("h")[0], scipy.array([self.h], dtype=scipy.float32))
        drv.memcpy_htod(self.mod_K.get_global("four_pi_gamma")[0],
                        scipy.array([4.*pi*self.gamma], dtype=scipy.float32))
        y = sqrt(self.gamma)/self.h
        drv.memcpy_htod(self.mod_K.get_global("y")[0], scipy.array([y], dtype=scipy.float32))
        n = scipy.arange(1, order+1, dtype=scipy.float32)
        n2 = n**2
        ex = exp(-n2/4.)
        pre_s2 = ex*cosh(n*y)
        pre_s3 = ex*n*sinh(n*y)
        drv.memcpy_htod(self.mod_K.get_global("n2")[0], n2)
        drv.memcpy_htod(self.mod_K.get_global("pre_s1")[0], ex)
        drv.memcpy_htod(self.mod_K.get_global("pre_s2")[0], pre_s2)
        drv.memcpy_htod(self.mod_K.get_global("pre_s3")[0], pre_s3)
        drv.memcpy_htod(self.mod_K.get_global("cos_phi")[0], cos(angles).astype(scipy.float32))
        drv.memcpy_htod(self.mod_K.get_global("sin_phi")[0], sin(angles).astype(scipy.float32))

    def K(self, Q, P, phix):
        quadratures = phix[:,1].copy()
        N_phix = phix.shape[0]
        Nx = Q.shape[0]
        Ny = int(floor(N_phix / 1024.))
        K = scipy.empty((Nx,), dtype=scipy.float32)
        Kb = drv.mem_alloc(4*Ny*Nx)
        Q_gpu = drv.to_device(Q)
        P_gpu = drv.to_device(P)
        self.K_gpu(drv.In(quadratures), Q_gpu, P_gpu, Kb,
                   block=(1, 1024, 1), grid=(Nx, Ny), shared=1024*4)
        self.reduction_gpu(Kb, drv.Out(K), block=(1, Ny, 1), grid=(Nx, 1), shared=Ny*4)
        return K/self.L

    def reconstruct_wigner(self, phix, Nq, Np):
        q_mean, p_mean, s_max = estimate_position_from_quadratures(self.eta, self.angles, phix[:,1])
        q, p, Q, P = build_mesh(q_mean, p_mean, s_max, Nq, Np)
        W = self.K(Q.ravel(), P.ravel(), phix)
        return q_mean, p_mean, Q, P, W.reshape(Q.shape)


def reconstruct_all_wigners(args):
    with h5py.File(args.filename, "r+") as h5:
        q_ds, p_ds, Q_ds, P_ds, W_ds = setup_reconstructions_group(h5, args.Nq, args.Np, args.force)
        Nsteps = h5["Quadratures"].shape[0]
        L = h5["Quadratures"].shape[1]
        no_angles, no_pulses = 100, 800
        angles = h5["Quadratures"][0,:,0].reshape(no_angles, no_pulses)[:,0].copy()
        calculator = CudaCalculator(args.eta, args.beta, L, angles, no_pulses, order=5)
        R = partial(calculator.reconstruct_wigner, Nq=args.Nq, Np=args.Np)
        start = time.time()
        for i, (q, p, Q, P, W) in enumerate(itertools.imap(R, h5["Quadratures"][:].astype(scipy.float32))):
            q_ds[i] = q
            p_ds[i] = p
            Q_ds[i,:,:] = Q
            P_ds[i,:,:] = P
            W_ds[i,:,:] = W
            elapsed = time.time()-start
            part = float(i)/Nsteps
            if part>0:
                eta = int(elapsed/part)
            else:
                eta = 0
            sys.stderr.write("\r{0:7.2%} (Elapsed: {1}, ETA: {2})".format(part,
                                                                          datetime.timedelta(seconds=int(elapsed)),
                                                                          datetime.timedelta(seconds=eta)))
        sys.stderr.write("\n")
    drv.stop_profiler()
