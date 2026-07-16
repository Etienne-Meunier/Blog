from dataclasses import dataclass
import torch
from einops import einsum as es, rearrange

@dataclass
class Dynamics :
    """
        Represent a set of j idenpendent dynamics
        each acting (in parallel, independently)
        on a state of dimension l.
    """
    j : int
    l : int

    def propagate(self, t0, z0, t) :
        """
            Propagate the points starting from (t0 and z0)
            using the dynamic and return their value in t

            Params :
                t0 (b) : reference time
                z0 (b j l) : reference state
                t (b k) : time to sample

            Return :
                z (b k j l) : propagated states
        """
        pass

@dataclass
class Dense(Dynamics) :
    """
        Each dynamic is represented as a dense l,l matrix
    """

    def __post_init__(self) :
        self.A = torch.randn(self.j, self.l, self.l)

    def propagate(self, t0, z0, t) :
        dts = t - t0[:, None] #  (b, k)
        Adts = es(dts, self.A, 'b k, j v l -> b k j v l')
        eA = torch.matrix_exp(Adts)  # (b k j v l)
        z = es(eA, z0, 'b k j v l, b j l -> b k j v')
        return z

@dataclass
class ScalingRotation(Dynamics) :
    """
        The dynamic is given as a l,l matrix
        with 2x2 blocks representing a scaling an a rotation.
        parameters are just factor of this scaling and rotation,
        note in the case where l=2 this correspond to "pure" koopman approach
        with complex eigenfunctions
    """
    def __post_init__(self) :
        assert self.l % 2 == 0, 'l needs to be even for scaling rotation dynamics'
        l2 = self.l // 2
        self.gamma, self.omega = torch.randn(self.j, l2), torch.randn(self.j, l2)
        self.J = torch.tensor([[0., -1.], [1., 0.]])
        self.I = torch.eye(2)

    def matrix_exp_Adt(self, dt) :
        """
            Compute e^{A dt} per 2x2 block

            Params :
                dt (b k) gaps we want to compute
            Returns :
                blocks_exp (b k j l/2 2 2) : matrix exponential per block

        """
        arg_g = es(dt, self.gamma, 'b k, j l -> b k j l')  # (b k j l/2)
        arg_o = es(dt, self.omega, 'b k, j l -> b k j l')  # (b k j l/2)
        eg, co, so = torch.exp(arg_g), torch.cos(arg_o), torch.sin(arg_o)  # (b k j l/2)
        rot = es(co, self.I, 'b k j l, d c -> b k j l d c') \
            + es(so, self.J, 'b k j l, d c -> b k j l d c')  # (b k j l/2 2 2)
        blocks_exp = es(eg, rot, 'b k j l, b k j l d c -> b k j l d c')  # (b k j l/2 2 2)
        return blocks_exp

    def propagate(self, t0, z0, t) :
        dts = t - t0[:, None] #  (b, k)
        z0_f = rearrange(z0, 'b j (ll c) -> b j ll c', c=2)
        eA = self.matrix_exp_Adt(dts)  # (b k j l/2 2 2)
        z_f = es(eA, z0_f, 'b k j ll d c, b j ll c -> b k j ll d')
        z = rearrange(z_f, 'b k j ll d -> b k j (ll d)')
        return z



# sequences 
t0, x0 <- sequence 
t, x <- sequence
z0 = encoder(x0)
tilde_z = dynamic.propagate(t0, z0, t)
tilde_x = decoder(tilde_z)
L = (x - tilde_x).sum()
backprop()