import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

lx,ly=1.0,1.0  # キャビティの横縦の幅
nx,ny=50,50    # 格子点数（全領域は0≦i≦nx, 0≦j≦ny）
dt=0.01        # Δt
nend=2000     # 終了時間ステップ
Re=200         # Reynolds数
plot_interval=100  # 何ステップおきにプロットするか

dx=lx/nx       # Δx
dy=ly/ny       # Δy
N=(nx-1)*(ny-1)  # 内部格子点数

psi = np.zeros((nx+1,ny+1))        # 流れ関数
omega = np.zeros((nx+1,ny+1))      # 渦度
new_omega = np.zeros((nx+1,ny+1))  # 次の時刻の渦度

def idx(i,j):  # 内部格子点 (i,j) をベクトル成分番号に変換する関数
    return (i-1)*(ny-1)+(j-1)

A=sp.lil_matrix((N, N))  # Poisson方程式の係数行列
b=np.zeros(N)  # Poisson方程式の右辺

for i in range(1,nx):  # ψのPoisson方程式の係数行列A
    for j in range(1,ny):
        k=idx(i,j)
        A[k,k]=2.0/dx**2+2.0/dy**2  # 対角成分
        if i>1:    A[k,idx(i-1,j)]=-1.0/dx**2  # 左
        if i<nx-1: A[k,idx(i+1,j)]=-1.0/dx**2  # 右
        if j>1:    A[k,idx(i,j-1)]=-1.0/dy**2  # 下
        if j<ny-1: A[k,idx(i,j+1)]=-1.0/dy**2  # 上
A = A.tocsr()

# グラフィックス用
xi,yi=np.meshgrid(np.linspace(0,lx,nx+1),np.linspace(0,ly,ny+1),indexing='ij')
plt.ion()

for n in range(0,nend+1):  # 時間ステップのループ（メインループ）
    for i in range(1,nx):  # ωの上下壁の境界条件
            omega[i,0]=-2*psi[i,1]/dy**2
            omega[i,ny]=-2*(psi[i,ny-1]+dy)/dy**2

    for j in range(1,ny):  # ωの左右壁の境界条件
            omega[0,j]=-2*psi[1,j]/dx**2
            omega[nx,j]=-2*psi[nx-1,j]/dx**2

    for i in range(1,nx):  # ωの時間発展
        for j in range(1,ny):
            new_omega[i,j]=omega[i,j]+dt*( \
              -(psi[i,j+1]-psi[i,j-1])*(omega[i+1,j]-omega[i-1,j])/4/dx/dy \
              +(psi[i+1,j]-psi[i-1,j])*(omega[i,j+1]-omega[i,j-1])/4/dx/dy \
              +((omega[i+1,j]-2*omega[i,j]+omega[i-1,j])/dx**2 \
               +(omega[i,j+1]-2*omega[i,j]+omega[i,j-1])/dy**2)/Re)

    for i in range(1,nx):  # new_omegaをomegaに戻す
        for j in range(1,ny):
            omega[i,j]=new_omega[i,j]

    for i in range(1,nx):  # ψのPoisson方程式の右辺
        for j in range(1,ny):
            k=idx(i,j)
            b[k]=omega[i,j]

    sol = spla.spsolve(A,b) # Ax=bを解く

    for i in range(1,nx):  # 2次元配列に戻す
        for j in range(1,ny):
            psi[i,j]=sol[idx(i,j)]  # psiの引数に注意!

    if n%plot_interval==0:
        max_psi=np.max(psi)  # 逆渦も出るように等高線の値の配列を加工
        min_psi=np.min(psi)
        if max_psi>0 and min_psi<0:
           levels=np.concatenate([
               np.linspace(min_psi,0.0,30,endpoint=False),
               np.linspace(0.0,max_psi,10)
           ])
        else:
           levels=np.linspace(min_psi,max_psi,20)

        plt.clf()  # グラフィックス
        cs=plt.contour(xi,yi,psi,levels,colors='black',linestyles='solid')
        ax=plt.gca()
        ax.set_aspect('equal',adjustable='box')
        ax.set_xlim(0.0,lx)
        ax.set_ylim(0.0,ly)
        plt.title(f"Re={Re},  time={n*dt}")
        plt.pause(0.1)

plt.ioff()  # 終了処理
plt.show()
