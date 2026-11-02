"""Smooth no-slip initial vortices"""
import numpy as np
import ufl

PRESETS = {
    "same_sign": ((1.0,0.35,0.5,0.12),(1.0,0.65,0.5,0.12)),
    "mixed_sign": ((1.0,0.35,0.35,0.10),(-1.0,0.65,0.35,0.10),
                   (-1.0,0.35,0.65,0.10),(1.0,0.65,0.65,0.10)),
}


def resolve_bumps(preset="same_sign"):
    """Each bump is (amplitude, center_x, center_y, width)."""
    
    if preset not in PRESETS:
        raise ValueError(f"Choose a preset from {list(PRESETS)}")
    bumps=PRESETS[preset]
    values=np.asarray(bumps,dtype=float)
    return tuple(tuple(float(x) for x in row) for row in values)


def sampled_velocity(x,y,bumps):
    """Independent NumPy derivatives used to estimate the continuous peak speed."""
    x,y=np.broadcast_arrays(np.asarray(x,dtype=float),np.asarray(y,dtype=float))
    sx,sy=np.sin(np.pi*x),np.sin(np.pi*y)
    envelope=sx*sx*sy*sy
    envelope_x=np.pi*np.sin(2*np.pi*x)*sy*sy
    envelope_y=np.pi*np.sin(2*np.pi*y)*sx*sx
    F,Fx,Fy=(np.zeros_like(x) for _ in range(3))
    for A,cx,cy,width in bumps:
        bump=A*np.exp(-((x-cx)**2+(y-cy)**2)/(2*width**2))
        F+=bump; Fx-=bump*(x-cx)/width**2; Fy-=bump*(y-cy)/width**2
    return np.stack((envelope_y*F+envelope*Fy,-(envelope_x*F+envelope*Fx)),axis=-1)


def normalization_scale(bumps,points=201):
    """Mesh-independent grid estimate of 1/||initial velocity||_Linfinity."""
    grid=np.linspace(0,1,points)
    x,y=np.meshgrid(grid,grid,indexing="xy")
    peak=float(np.linalg.norm(sampled_velocity(x,y,bumps),axis=-1).max())
    if not np.isfinite(peak) or peak<=1e-14:
        raise ValueError("Initial profile must have nonzero sampled speed")
    return 1/peak


def initial_velocity(domain,bumps,scale):
    """Symbolically differentiate the enveloped Gaussian streamfunction."""
    x=ufl.SpatialCoordinate(domain)
    F=0
    for A,cx,cy,width in bumps:
        F+=A*ufl.exp(-((x[0]-cx)**2+(x[1]-cy)**2)/(2*width**2))
    psi=scale*ufl.sin(np.pi*x[0])**2*ufl.sin(np.pi*x[1])**2*F
    return ufl.as_vector((psi.dx(1),-psi.dx(0)))

