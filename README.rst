Network Inference in Stochastic Systems
=======================================

Introduction
-----------------------------
We developed a data-driven approach for network inference in stochastic systems, Free Energy Minimazation (FEM). From data comprising of configurations of variables, we determine the interactions between them in order to infer a predictive stochastic model. FEM outperforms other existing methods such as variants of mean field approximations and Maximum Likelihood Estimation (MLE), especially in the regimes of large coupling variability and small sample sizes. Besides better performance, FEM is parameter-free and significantly faster than MLE.

Interactive notebook
-----------------------------
Use Binder to run our code online. You are welcome to change the parameters and edit the jupyter notebooks as you want. 

.. image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/nihcompmed/network-inference/master?filepath=sphinx%2Fcodesource

Links
----------------------------
Code Documentation
    https://nihcompmed.github.io/network-inference

Code Source
    https://github.com/nihcompmed/network-inference

Reference
----------------------------
Danh-Tai Hoang, Juyong Song, Vipul Periwal, and Junghyo Jo, "Causality inference in stochastic systems from neurons to currencies: Profiting from small sample size", `arXiv:1705.06384 <https://arxiv.org/abs/1705.06384>`_.
