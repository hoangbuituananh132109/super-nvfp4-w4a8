import numpy as np
rng=np.random.default_rng(0)
E2M1=np.array([0,0.5,1,1.5,2,3,4,6],np.float64)
def rne_grid(x,grid):
    # round to nearest in a sorted nonneg grid (ties to even index ~ RNE for these grids)
    a=np.abs(x); idx=np.searchsorted(grid,a); idx=np.clip(idx,1,len(grid)-1)
    lo=grid[idx-1]; hi=grid[idx]; pick_hi=(a-lo)>(hi-a)
    tie=(a-lo)==(hi-a); pick_hi=np.where(tie,(idx%2==0),pick_hi)
    return np.sign(x)*np.where(pick_hi,hi,lo)
# full E4M3 (fn) positive grid
e4=[ (m/8)*2**-6 for m in range(8)]
for e in range(1,16):
    for m in range(8):
        if e==15 and m==7: continue
        e4.append((1+m/8)*2**(e-7))
E4M3=np.array(sorted(set(e4)))
def fp8(x): return rne_grid(np.clip(x,-448,448),E4M3)
def nvfp4_quant(w):  # ModelOpt-style: per-tensor global, per-16 E4M3 scale
    N,K=w.shape; g=w.reshape(N,K//16,16)
    amax=np.abs(g).max(-1)
    gs=np.abs(w).max()/(6*448)          # weight_scale_2
    s=fp8(amax/6/gs)                     # E4M3 block scale (stored)
    s=np.where(s==0,E4M3[1],s)
    q=rne_grid(g/(s[...,None]*gs),E2M1)
    return q,s,gs
def mxfp4_quant(w):
    N,K=w.shape; g=w.reshape(N,K//32,32); amax=np.abs(g).max(-1)
    e=np.floor(np.log2(np.maximum(amax,1e-30)))-2  # OCP: 2^(floor(log2 amax)-emax_elem)
    s=2.0**e; q=rne_grid(g/s[...,None],E2M1); q=np.clip(q,-6,6)
    return (q*s[...,None]).reshape(N,K)
def rel(a,b): return np.linalg.norm(a-b)/np.linalg.norm(b)
for name,gen in [('gauss',lambda:rng.standard_normal((1024,2688))),
                 ('t3+chan',lambda:rng.standard_t(3,(1024,2688))*np.exp(rng.normal(0,0.7,(1024,1))))]:
    w=gen()*0.02
    q,s,gs=nvfp4_quant(w); N,K=w.shape
    w_nv=(q*s[...,None]*gs).reshape(N,K)
    # option A: per-row power-of-two residual; FP8 = fp8(q*s/r)
    smax=s.max(1); r=2.0**np.ceil(np.log2(6*smax/448))
    wA=fp8(q*(s/r[:,None])[...,None]).reshape(N,K)*r[:,None]*gs
    # option B: per-row exact-max residual
    rB=6*smax/448; wB=fp8(q*(s/rB[:,None])[...,None]).reshape(N,K)*rB[:,None]*gs
    # option P: per-tensor only (no residual): s already <=448, q*s up to 2688 -> scale by 1/8 (pow2)
    wP=fp8(q*(s/8)[...,None]).reshape(N,K)*8*gs
    w_mx=mxfp4_quant(w); w_f8=fp8(w/(np.abs(w).max(1,keepdims=True)/448))*(np.abs(w).max(1,keepdims=True)/448)
    print(name, f"NVFP4 vs BF16 {rel(w_nv,w):.4f} | MXFP4 vs BF16 {rel(w_mx,w):.4f} | FP8row vs BF16 {rel(w_f8,w):.4f}")
    print(name, f"  NV->FP8 extra (vs w_nv): A pow2-row {rel(wA,w_nv):.4f}  B exact-row {rel(wB,w_nv):.4f}  P per-tensor {rel(wP,w_nv):.4f}")
    print(name, f"  total vs BF16: A {rel(wA,w):.4f}  B {rel(wB,w):.4f}  P {rel(wP,w):.4f}; frac elems changed A {(wA!=w_nv).mean():.3f}")
    # GEMM-level with FP8 activations per token
    x=rng.standard_normal((256,2688))*np.exp(rng.normal(0,1,(256,1)))
    xs=np.abs(x).max(1,keepdims=True)/448; x8=fp8(x/xs)*xs
    y0=x@w.T; y1=x@w_nv.T; y2=x8@wA.T
    print(name, f"  GEMM rel: R1-R0 {rel(y1,y0):.4f}  R2-R1 {rel(y2,y1):.4f}  R2-R0 {rel(y2,y0):.4f}  (A8 alone {rel(x8@w_nv.T,y1):.4f})")
