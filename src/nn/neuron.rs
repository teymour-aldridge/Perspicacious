use super::activations::activations;

pub fn dot_product(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    // Calculate the dot product of two vectors.
    assert_eq!(a.len(), b.len());
    let mut product: f64 = 0 as f64;
    for i in 0..a.len() {
        product += a[i] * b[i];
    }
    product
}

pub struct Neuron {
    pub weights: Vec<f64>,
    lr: f64,
    pub activation_function: activations::ActivationFunction,
    pub output_cache: f64,
    pub delta: f64,
}

impl Neuron {
    pub fn new(weights: &Vec<f64>, lr: &f64, activation_function: activations::ActivationFunction) -> Neuron {

        Neuron {
            weights: weights.to_vec(),
            lr: lr.to_degrees(),
            activation_function,
            output_cache: 0 as f64,
            delta: 0.0,
        }
    }
    pub fn output(&mut self, inputs: &Vec<f64>) -> f64 {
        self.output_cache = dot_product(inputs, &self.weights);
        (self.activation_function.function)(&self.output_cache)
    }
}