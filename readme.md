# Mamba-S4

***On this project i'm attempting to recreate S4 and Mamba from scratch in Pytorch***


## Scope

The project is just tinkering around with these architectures specially on an ongoing issue that is the scan that they use.

The method i'm trying to implement is the Heisen Sequence [Paper](https://arxiv.org/abs/2311.06281) which makes an efficient way to compute the prefix sum on equations with the form of : 

$$ x_{t}=a_{t}x_{t−1}+b_{t} $$

which is the same type of equation present on State-space representations.

### State-Space

The state-space representation is has a lot of applications in engineering in the field of control theory, this models are foundations of understanding the various states of complex systems, e.g [Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter).


As of January it seems that Pytorch doesn't have this algorithm, which could be really beneficial for state-space models.

### S4

In order to check if the algorithm is working as expected i'll be recreating this interesting content:

[The annotated S4](https://srush.github.io/annotated-s4/)
In this post S4 is implemented in JAX which has an implementation of [Prefix sum](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.associative_scan.html), from Blelloch's.

### Mamba

Once the algorithm is verified with S4 the mamba implementation will proceed.


## Theory

Check the following papers in order to understand the theory:
- Efficient Parallelization of a Ubiquitous Sequential Computation  [Paper](https://arxiv.org/abs/2311.06281) 
- Efficiently Modeling Long Sequences with Structured State Spaces [Paper](https://arxiv.org/abs/2111.00396)
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces [Paper](https://arxiv.org/abs/2312.00752)

### State-space 

- State Space Representation [Wiki](https://en.wikipedia.org/wiki/State-space_representation)
- Examples of State Space Representation in Control Theory [ECE 486 Control Systems- illinois](https://courses.engr.illinois.edu/ece486/fa2023/documentation/handbook/lec02.html)