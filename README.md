# Neuromorphic-Hebbian-Learning
**Simple Python simulation of a single LIF neuron learning a digit pattern using Hebbian learning.**

A single LIF neuron is initialized with **random** weights, a data-set containing different examples of a "0" digit is given as input to the neuron and the **weights are updated** at each time step following **Oja's rule** which is a **stable** type of Hebbian learning rule. Over time, the synaptic weights of the neuron converge to the average of the data-set. The Oja's rule is defined as follows:

h<sub>&theta;</sub>(x) = &theta;<sub>o</sub> x + &theta;<sub>1</sub>x 
