.. swidi documentation master file, created by
   sphinx-quickstart on Wed Feb 18 19:31:38 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SwiDi - Simulation of stochastic differential equations with Markovian switching
================================================================================

SwiDi is a software framework written in `Python <https://www.python.org>`_. Its purpose is to provide an environment
for the numerical simulation of `stochastic differential equations
<http://en.wikipedia.org/wiki/Stochastic_differential_equation>`_ with `Markovian
<http://en.wikipedia.org/wiki/Markov_chain>`_ switching. The code is freely available under

 `<https://github.com/michaelschaefer/swidi/>`_

A short introduction of the underlying mathematical model is
presented in the chapter `Mathematical model`_.

API Documentation
=================

.. toctree::
    :maxdepth: 2
    :glob:


Mathematical model
==================

Let :math:`\mathcal{M} \subset \mathbb{N}` be a finite set with :math:`|\mathcal{M}| = m` and :math:`T > 0`. We introduce
a *drift* function :math:`f : \mathbb{R}^n \times \mathcal{M} \times [0, T] \to \mathbb{R}^n`, a *diffusion* function
:math:`g : \mathbb{R}^n \times \mathcal{M} \times [0,T] \to \mathbb{R}^{n \times m}`, a :math:`m`-dimensional
*Brownia motion* :math:`W : [0, T] \to \mathbb{R}^m` and a *Markovian control parameter* :math:`\alpha : [0, T] \to
\mathcal{M}`. We seek a *state* :math:`X : [0, T] \to \mathbb{R}^n` satisfying

.. math::

    \textnormal{d}X(t) &= f(X(t), \alpha(t), t) \textnormal{d}t + g(X(t), \alpha(t), t) \textnormal{d}W(t), \\
    X(0) &= x_0

for an *initial condition* :math:`x_0 \in \mathbb{R}^n`. The control parameter :math:`\alpha` is a stochastic process
given by

.. math::

    \mathbb{P}(\alpha(t + \Delta) = j \mid \alpha(t) = i, (X(s), \alpha(s)) \forall s < t) = q_{ij}(X(t)) \Delta
    + o(\Delta),

where the *generator matrix* :math:`Q(x) := (q_{ij}(x))_{i,j=1}^m` has the following properties:

.. math::

    q_{ij}(x) &\geq 0 \quad \forall i, j \in \mathcal{M} \text{ with } i \neq j,\\
    \sum_{j=1}^m q_{ij}(x) &= 0 \quad \forall i \in \mathcal{M}

For further details also about the numerical implementation, we refer to [MYY07]_.


.. [MYY07] Mao, X., Yuan, C. & Yin, G. (2007). Approximations of Euler–Maruyama type for stochastic differential
   equations with Markovian switching, under non-Lipschitz conditions. *Journal of Computational and Applied
   Mathematics, 205(2)*, 936-948. doi: `10.1016/j.cam.2006.01.052 <http://dx.doi.org/10.1016/j.cam.2006.01.052>`_


