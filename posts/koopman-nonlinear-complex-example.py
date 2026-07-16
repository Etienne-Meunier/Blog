


import torch
from torchdiffeq import odeint
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange

@dataclass
class ComplexPoint:
    a: float
    b: float
    c : float
    w : float
    

    def f(self, t, x):
        """x (..., 2) -> dx/dt (..., 2)"""
        x1, x2 = x[..., 0], x[..., 1]
        xp1 = self.a * x1 - self.w * x2 + self.w * self.c * x1**2

        xp2 = (
            self.w * x1
            + self.a * x2
            + self.a * self.c * x1**2
            - 2 * self.c * self.w * x1 * x2
            + 2 * self.c**2 * self.w * x1**3
        )

        return torch.stack([xp1, xp2], dim=-1)

fp = ComplexPoint(1.0, 2.0, 3.0, 4.0)

x0 = torch.tensor([1., 2.])
t_int = torch.linspace(0, 2, 100)
x_gt = odeint(fp.f, x0, t_int)

# Koopman eigenfunctions


#   psi_+(x) = x1 + i*(x2 - c*x1^2),   eigenvalue  a + i*w
#   psi_-(x) = x1 - i*(x2 - c*x1^2),   eigenvalue  a - i*w

# True eigenfunction
def psi(x):
    x1, x2 = x[...,0], x[...,1]
    return torch.complex(x1, x2 - fp.c*x1**2)



# Test one (not an eigenfunction)
def f(x):
    x1, x2 = x[...,0], x[...,1]
    return torch.complex(x1 + x2, x1- x2)


fig, axs = plt.subplots(1, 2, figsize=(5, 4))
axs[0].set_title('Physical coordinates')
axs[0].scatter(*x_gt.T, c=t_int)
axs[0].set_xlabel(r"$x_1$")
axs[0].set_ylabel(r"$x_2$")

axs[1].set_title('Koopman coordinates')
axs[1].scatter(psi(x_gt).real, psi(x_gt).imag, c=t_int)
axs[1].set_xlabel(r"$real$")
axs[1].set_ylabel(r"$imag$")

plt.show()

# Here we need to adapt our technique to test if the projected trajectory
# Follows a linear dynamic in complex space

psi_gt = psi(x_gt)

psi_w = psi_gt.angle()
psi_w_uw = np.unwrap(psi_w)


fig, axs = plt.subplots(1, 2, figsize=(9, 3.5), sharex=True)
for ax, y, title in zip(axs, (psi_w, psi_w_uw), ('Angles', 'Unwrapped angles')):
    ax.plot(t_int, y, color='0.7', lw=0.8, zorder=1)
    ax.scatter(t_int, y, c=t_int, cmap='viridis', s=3, zorder=2)
    ax.set_title(title)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\psi$ angle [rad]")

fig.tight_layout()
plt.show()

psi_r = psi_gt.abs()

a = torch.log(psi_r[1:] / psi_r[0]) / t_int[1:]
b = (psi_w_uw[1:] - psi_w_uw[0]) / t_int[1:]



#
gamma, omega = a.mean(), b.mean()

psi_0 = psi(x_gt[0])
psi_0_r, psi_0_w = psi_0.abs(), psi_0.angle()

tilde_psi_t_r, tilde_t_psi_w = torch.exp(gamma*t_int)*psi_0_r, psi_0_w + omega * t_int
tilde_psi_t = torch.polar(tilde_psi_t_r, tilde_t_psi_w)

(psi(x_gt) - tilde_psi_t).abs()

fig, axs = plt.subplots(1, 2)

axs[0].scatter(psi(x_gt).real,psi(x_gt).imag)
axs[1].scatter(tilde_psi_t.real,tilde_psi_t.imag)

tilde_psi_t

# Reconstruction 

xs = torch.randn(10, 2)
Psi = lambda x : torch.stack([psi(x), psi(x)**2, psi(x)*psi(x).conj()], dim=-1) # (n, J)

def complex_to_component(z) :
    """
        z (n, J) -> c (n (J, 2))

    Take a complex tensor and return a real tensor with the two components stacked 
    """
    r = torch.view_as_real(z)
    r_flat = rearrange(r, 'n J c -> n (J c)')
    return r_flat

V = torch.linalg.lstsq(complex_to_component(Psi(xs)), xs).solution.T  

tilde_x_gt = complex_to_component(Psi(x_gt))@V.T 

fig, axs = plt.subplots(1, 2)

axs[0].scatter(*x_gt.T)
axs[1].scatter(*tilde_x_gt.T)

tilde_psi_t


# Forecast : 

Gammas = torch.tensor([gamma, 2*gamma, 2*gamma]) # (J)
Omegas = torch.tensor([omega, 2*omega, 0]) # (J)


psi_0 = Psi(x_gt[0])
psi_0_r, psi_0_w = psi_0.abs(), psi_0.angle() #(J), (J)

psi_r = torch.exp(Gammas[None]* t_int[:, None])*psi_0_r[None] # (T, J)
psi_w = psi_0_w[None] + Omegas*t_int[:, None] # (T, J)

psi_t = complex_to_component(torch.polar(psi_r, psi_w)) # (T, (2*J))

tilde_x_gt = psi_t@V.T

fig, axs = plt.subplots(1, 2)

axs[0].scatter(*x_gt.T, c=t_int)
axs[1].scatter(*tilde_x_gt.T, c=t_int)

tilde_psi_t

