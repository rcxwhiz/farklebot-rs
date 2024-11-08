use super::activation_function::{self, ActivationFunction};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerTopology {
    pub n_neurons: usize,
    pub activation_function: ActivationFunction
}

impl LayerTopology {
    pub fn new(n_neurons: usize, activation_function: ActivationFunction) -> Self {
        Self { n_neurons, activation_function }
    }
}
