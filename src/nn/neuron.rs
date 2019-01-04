use super::activation_functions;

pub fn dot_product(a: Vec<f64>, b: Vec<f64>) -> f64 {
    // Calculate the dot product of two vectors.
    asserteq!(a.len(), b.len());
    let mut product: f64;
    for i in 0..a.len() {
        product += a[i] * b[i];
    }
    product
}


pub struct Neuron {
    weights: Vec<f64>,
    lr: f64,
    activation_function: activation_functions::ActivationFunction,
    output_cache: f64,
}

impl Neuron {
    fn new(weights: Vec<f64>, lr: f64, activation_function: activation_functions::ActivationFunction) -> Neuron {
        Neuron {
            weights,
            lr,
            activation_function,
            output_cache: 0 as f64,
        }
    }
    fn output(&mut self, inputs: Vec<f64>) -> f64 {
        self.output_cache = dot_product(inputs, self.weights);
        self.activation_function.function(self.output_cache)
    }
}