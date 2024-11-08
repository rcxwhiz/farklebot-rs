use ndarray::{Array2, Axis};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;


#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActivationFunction {
    Linear,
    Relu,
    Sigmoid,
    Softmax,
}

impl ActivationFunction {
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        match self {
            ActivationFunction::Linear => input.clone(),
            ActivationFunction::Relu => input.mapv(|x| if x < 0.0 { 0.0 } else { x }),
            ActivationFunction::Sigmoid => input.mapv(|x| 1.0 / (1.0 + (-x).exp())),
            ActivationFunction::Softmax => {
                let max_values = input.fold_axis(Axis(1), f32::MIN, |&acc, &x| acc.max(x));
                let exp_values = input - &max_values.insert_axis(Axis(1));
                let exp_values = exp_values.mapv(|x| x.exp());
                let sum_exp = exp_values.sum_axis(Axis(1)).insert_axis(Axis(1));

                exp_values / &sum_exp
            }
        }
    }

    pub fn initial_weights(&self, n_inputs: usize, n_neurons: usize) -> Array2<f32> {
        match self {
            ActivationFunction::Linear | ActivationFunction::Relu => {
                let dist = Uniform::new(-1.0, 1.0);
                Array2::random((n_neurons, n_inputs), dist) * (2.0 / n_inputs as f32).sqrt()
            }
            ActivationFunction::Sigmoid | ActivationFunction::Softmax => {
                let limit = (6.0 / (n_inputs + n_neurons) as f32).sqrt();
                let dist = Uniform::new(-limit, limit);
                Array2::random((n_neurons, n_inputs), dist)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_close(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    fn eq(a: &Array2<f32>, b: &Array2<f32>) -> bool {
        if a.shape() != b.shape() {
            return false;
        }
        for (c, d) in a.iter().zip(b.iter()) {
            if !is_close(*c, *d, 1e-5) {
                return false;
            }
        }
        true
    }

    #[test]
    fn linear() {
        let input = Array2::from_shape_vec((1, 2), vec![1.0, -2.0]).unwrap();
        let expected_result = Array2::from_shape_vec((1, 2), vec![1.0, -2.0]).unwrap();
        let result = ActivationFunction::Linear.forward(&input);
        assert!(eq(&expected_result, &result))
    }

    #[test]
    fn relu() {
        let input = Array2::from_shape_vec((1, 2), vec![1.0, -2.0]).unwrap();
        let expected_result = Array2::from_shape_vec((1, 2), vec![1.0, 0.0]).unwrap();
        let result = ActivationFunction::Relu.forward(&input);
        assert!(eq(&expected_result, &result))
    }

    #[test]
    fn sigmoid() {
        let input = Array2::from_shape_vec((1, 2), vec![1.0, -2.0]).unwrap();
        let expected_result = Array2::from_shape_vec((1, 2), vec![0.7310585786, 0.119202922]).unwrap();
        let result = ActivationFunction::Sigmoid.forward(&input);
        assert!(eq(&expected_result, &result))
    }

    #[test]
    fn softmax() {
        let input = Array2::from_shape_vec((1, 2), vec![1.0, -2.0]).unwrap();
        let expected_result = Array2::from_shape_vec((1, 2), vec![0.952574, 0.047426]).unwrap();
        let result = ActivationFunction::Softmax.forward(&input);
        assert!(eq(&expected_result, &result))
    }
}