import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

lx,ly=1.0,1.0  # 領域の横縦の幅
nx,ny=50,50    # 格子点数（全領域は0≦i≦nx, 0≦j≦ny）

dx=lx/nx       # Δx
dy=ly/ny       # Δy
N=(nx-1)*(ny-1)  # 内部格子点数

u = np.zeros((nx+1,ny+1))        # Δu=-fのu

def idx(i,j):  # 内部格子点 (i,j) をベクトル成分番号に変換する関数
    return (i-1)*(ny-1)+(j-1)

def f(i,j):    # Δu=-fのf
    return 2*(i*dx*(1-i*dx)+j*dy*(1-j*dy))

A=sp.lil_matrix((N, N))  # Poisson方程式の係数行列
b=np.zeros(N)  # Poisson方程式の右辺

for i in range(1,nx):  # uのPoisson方程式の係数行列A
    for j in range(1,ny):
        k=idx(i,j)
        A[k,k]=2.0/dx**2+2.0/dy**2  # 対角成分
        if i>1:    A[k,idx(i-1,j)]=-1.0/dx**2  # 左
        if i<nx-1: A[k,idx(i+1,j)]=-1.0/dx**2  # 右
        if j>1:    A[k,idx(i,j-1)]=-1.0/dy**2  # 下
        if j<ny-1: A[k,idx(i,j+1)]=-1.0/dy**2  # 上
A = A.tocsr()

for i in range(0,nx+1):  # 一応 u の上下境界条件
        u[i,0]=0.0
        u[i,ny]=0.0

for j in range(0,ny+1):  # 一応 u の左右境界条件
        u[0,j]=0.0
        u[nx,j]=0.0

for i in range(1,nx):  # Poisson方程式の右辺
    for j in range(1,ny):
        k=idx(i,j)
        b[k]=f(i,j)  # 右辺

sol = spla.spsolve(A,b) # Ax=bを解く

for i in range(1,nx):  # 2次元配列に戻す
    for j in range(1,ny):
        u[i,j]=sol[idx(i,j)]  # psiの引数に注意!

# グラフィックス
xi,yi=np.meshgrid(np.linspace(0,lx,nx+1),np.linspace(0,ly,ny+1),indexing='ij')

levels=np.linspace(0.003,0.06,20)
cs=plt.contour(xi,yi,u,levels,colors='black',linestyles='solid')
plt.clabel(cs)  # 数値ラベル
ax=plt.gca()
ax.set_aspect('equal',adjustable='box')
ax.set_xlim(0.0,lx)
ax.set_ylim(0.0,ly)
plt.title('Poisson eq')
plt.show()
