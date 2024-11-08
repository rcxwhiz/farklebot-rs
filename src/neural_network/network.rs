use std::vec;

use ndarray::{Array1, Array2, Axis};
use crate::neural_network::layer::Layer;
use crate::neural_network::activation_function::ActivationFunction;
use crate::neural_network::layer_topology::LayerTopology;


#[derive(Clone, Debug, PartialEq)]
pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn new(layers: Vec<Layer>) -> Self {
        assert!(!layers.is_empty());
        layers.windows(2).for_each(|window| {
            assert_eq!(window[1].n_inputs(), window[0].n_neurons());
        });
        
        Self { layers }
    }

    pub fn random(topologies: &Vec<LayerTopology>) -> Self {
        assert!(topologies.len() > 1);

        let layers = topologies
            .windows(2)
            .map(|window| Layer::random(window[0].n_neurons, window[1].n_neurons, window[1].activation_function))
            .collect();

        Self::new(layers)
    }
    
    pub fn input_size(&self) -> usize {
        self.layers.first().unwrap().n_inputs()
    }
    
    pub fn output_size(&self) -> usize {
        self.layers.last().unwrap().n_neurons()
    }

    pub fn predict_single(&self, input: &Array1<f32>) -> Array1<f32> {
        let input_2d = input.clone().into_shape((1, input.len())).unwrap();

        let output_2d = self.predict(&input_2d);

        output_2d.index_axis(Axis(0), 0).to_owned()
    }

    pub fn predict(&self, input: &Array2<f32>) -> Array2<f32> {
        assert_eq!(input.nrows(), self.layers.first().unwrap().n_inputs());

        self.layers.iter().fold(input.clone(), |acc, layer| {
            layer.forward(&acc)
        })
    }

    pub fn topology(&self) -> Vec<LayerTopology> {
        let mut topology = vec![LayerTopology::new(self.layers.first().unwrap().n_inputs(), ActivationFunction::Linear)];
        for layer in &self.layers {
            topology.push(layer.topology());
        }
        topology
    }

    pub fn to_genome(&self) -> Vec<f32> {
        let mut genome = Vec::new();
        for layer in &self.layers {
            genome.extend(layer.weights.iter());
            genome.extend(layer.biases.iter());
        }
        genome
    }

    pub fn from_genome(genome: Vec<f32>, topology: &Vec<LayerTopology>) -> Self {
        let mut genome_iter = genome.into_iter();
        let mut network_layers = Vec::new();
        for window in topology.windows(2) {
            let input_size = window[0].n_neurons;
            let output_size = window[1].n_neurons;
            let activation_function = window[1].activation_function;

            let weights: Array2<f32> = Array2::from_shape_fn((output_size, input_size), |_| {
                genome_iter.next().unwrap()
            });
            let biases: Array1<f32> = Array1::from_shape_fn(output_size, |_| {
                genome_iter.next().unwrap()
            });

            let layer = Layer::new(weights, biases, activation_function);
            network_layers.push(layer);
        }

        Network::new(network_layers)
    }
}
