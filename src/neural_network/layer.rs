use ndarray::{Array1, Array2};

use crate::neural_network::activation_function::ActivationFunction;

use super::layer_topology::LayerTopology;

#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    pub weights: Array2<f32>,
    pub biases: Array1<f32>,
    pub activation_function: ActivationFunction,
}

impl Layer {
    pub fn new(weights: Array2<f32>, biases: Array1<f32>, activation_function: ActivationFunction) -> Self {
        assert_eq!(weights.nrows(), biases.len());
        
        Self { weights, biases, activation_function }
    }
    
    pub fn random(n_inputs: usize, n_neurons: usize, activation_function: ActivationFunction) -> Self {
        let weights = activation_function.initial_weights(n_inputs, n_neurons);
        let biases = Array1::zeros(n_neurons);

        Self::new(weights, biases, activation_function)
    }
    
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        assert_eq!(self.n_inputs(), input.ncols());
        
        let z = input.dot(&self.weights) + &self.biases;
        assert_eq!(self.n_neurons(), z.ncols());
        assert_eq!(input.nrows(), z.nrows());
        self.activation_function.forward(&z)
    }
    
    pub fn n_inputs(&self) -> usize {
        self.weights.ncols()
    }

    pub fn n_neurons(&self) -> usize {
        self.weights.nrows()
    }

    pub fn topology(&self) -> LayerTopology {
        LayerTopology::new(self.n_neurons(), self.activation_function)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_from_random() {
        let layer = Layer::random(2, 5, ActivationFunction::Linear);
        assert_eq!(layer.n_inputs(), 2);
        assert_eq!(layer.n_neurons(), 5);
        assert_eq!(layer.activation_function, ActivationFunction::Linear);
        assert_eq!(layer.weights.nrows(), 5);
        assert_eq!(layer.weights.ncols(), 2);
        assert_eq!(layer.biases.len(), 5);
    }

    #[test]
    fn to_topology() {
        let layer = Layer::random(6, 1, ActivationFunction::Relu);
        let topology = layer.topology();
        let expected_topology = LayerTopology::new(1, ActivationFunction::Relu);
        assert_eq!(topology, expected_topology);
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed\n  left: 2\n right: 3")]
    fn mismatched_shapes() {
        let _ = Layer::new(Array2::from_shape_vec((2, 2), vec![2.3, 4.5, 6.7, 8.9]).unwrap(), Array1::from_vec(vec![1.2, 3.4, 5.6]), ActivationFunction::Linear);
    }

    #[test]
    fn forward() {
        let layer = Layer::new(Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap(), Array1::from_vec(vec![5.0, 6.0]), ActivationFunction::Linear);
        let input: Array2<f32> = Array2::from_shape_vec((2, 2), vec![11.0, 12.0, 13.0, 13.0]).unwrap();
        let expected_output: Array2<f32> = Array2::from_shape_vec((2, 2), vec![52.0, 76.0, 57.0, 84.0]).unwrap();
        assert_eq!(layer.forward(&input), expected_output);
    }
}
