# Neuromorphic-Hebbian-Learning
**Simple Python simulation of a single LIF neuron learning a digit pattern using Hebbian learning.**

A single LIF neuron is initialized with **random** weights, a data-set containing different examples of a "0" digit is given as input to the neuron and the **weights are updated** at each time step following **Oja's rule** which is a **stable** type of Hebbian learning rule. Over time, the synaptic weights of the neuron converge to the average of the data-set. The Oja's rule is defined as follows:

**&Delta;</sub>w[n] = &alpha;[n]</sub> x[n] - &alpha;<sup>2</sup>[n] w[n]**

where **w[n]** is the synaptic weight vector, **&alpha;[n]** is the **filtered** spike train using a PSC kernel.

# Training Result

The code outputs the evolution of the weight vector over time. The weights have converged towards an image of a "0" digit. The pattern has thus been learned in an unsupervised way.

![Alt text](figures/w_evo.png?raw=true "Weight evolution in function of time")

# How to cite

If you use this work, please cite:

**Ali Safa. (2020). Hebbian Learning with Spiking Neuron. http://doi.org/10.5281/zenodo.3951585.**

```
@misc{ali_safa_2020_3951560,
  author       = {Ali Safa},
  title        = {{Hebbian Learning with Spiking Neuron}},
  month        = Aug,
  year         = 2020,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.3974508},
  howpublished = {\url{https://doi.org/10.5281/zenodo.3974508}}
}
```

