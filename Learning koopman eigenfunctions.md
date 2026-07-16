# Learning koopman eigenfunctions 

Here we get to the real deal ! Let's say we have a set of trajectory points which we don't know anything about except that they come from the same dynamical system and we want to learn to do inference for a new start point ! We could do auto-regressive models for instance but we want it to be fast for long term inference. In this setup we suppose that the points we have represent the "full state", they are not noisy or incomplete information and they determine uniquely the future. 

Unknown equation : 
$$
\frac{dx}{dt} = f(x)
$$
Data : 

$[(x_{k, t_1},  t_1), (x_{k, t_2},  t_2) \cdots (x_{k, t_I}, t_I)]$ for $k$ trajectories (different initial states)

The times $t_i$ are not uniformally sampled. 

Before starting let's remember what we learned in the previous notebooks working on simple case-studies : 

- Normally the reconstruction should be linear function of koopman eigenfunctions (which can be non-linear function of the data)
- If we multiply several koopman eigenfunctions together we obtain a new one 
- Koopman eigenfunctions evolve independently of each other, in parallel 
- They can be complex and we need complex eigenfunctions to represent rotations 
- Often koopman eigenfunctions and eigenvalues depend on the parameters of the dynamic so if they change in the dataset we need to take that in account
- We don't necessarly needs to evaluate the reconstruction ability on trajectory points only : it should be true for any valid point that the dynamic could output
- $g(x)$ doesn't have to be identity, we could train a koopman network for any function of $x$. 



The objective of learning a koopman eigenfunctions is : 

$$
\begin{aligned}
  \{\mathcal{E}, \mathcal{D}\} = \arg\min_{\mathcal{E},\, \mathcal{D}} \; &\mathbb{E}_{x \sim \rho}\Big[ \big\| g(x) - \mathcal{D}\big(\mathcal{E}(x)\big) \big\|^2 \Big] \\
  \text{s.t.}\quad &\exists\, A \;:\; \mathcal{E}\big(T(x,\Delta t)\big) = e^{A \Delta t}\,\mathcal{E}(x) \quad \text{for } \rho\text{-almost every } x
  \end{aligned}
$$
We take $\rho$ to be the distribution of states reachable by the dynamic: sampling an initial condition $x_0 \sim p_0$ and a time $t \sim p_T$. It is the density of  "realistic states".

Although this objective is hard to reach, especially the constraint part. Looking at the existing deep learning methods there is two common way to adress it  (DMD don't work too well on those data as the $t$ are not equally spaced and could be far appart) : 

1. <u>Unrolling : latent dynamic linear by construction</u> 

$$
\min_{A, \mathcal E, \mathcal D} ||x_{t+\Delta_t} - \mathcal D(e^{A \Delta_t}\mathcal E({x_t}))||_2^2
$$

For instance in : *"NIGO: Lyapunov-Stable Continuous-Time Neural  Operators for Long-Horizon PDE Dynamics"*

We can use either the exponential function or a neural ODE technique.

2. <u>Dual minimisation : lagrange formulation perspective on this</u> 
   $$
   \min_{A, \mathcal E, \mathcal D} ||x_{t} - \mathcal D(\mathcal E({x_t}))||_2^2 + \alpha ||\mathcal E(x_{t + \Delta_t}) - e^{A \Delta_t} \mathcal E({x_t})||_2^2
   $$

The advantage of that is that it doesn't require to do the full backprop through the prediction. 

For instance in : *"Deep learning for universal linear embeddings of nonlinear dynamics"*

Note : For those two methods we could make the reconstruction linear choosing a specific $\mathcal D$ and we could restreint the dynamic to be independent for each latent dymension constraining A. 



