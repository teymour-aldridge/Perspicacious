extern crate rand;

use std::option;
use std::vec::Vec;
use super::activations::activations;
use super::neuron;
use super::neuron::dot_product;

fn clone_vec<T: Clone>(vec: &[T]) -> Vec<T> {
    let mut newvec = vec.to_vec();
    newvec
}

pub enum PreviousLayer<Layer> {
    None,
    Some(Layer),
}

pub struct Layer {
    previous_layer: u64,
    pub neurons: Vec<neuron::Neuron>,
    output_cache: Vec<f64>,
}

impl Layer {
    fn new(previous_layer: u64, n_neurons: u64, lr: f64, activation_function: activations::ActivationFunction) -> Layer {
        // Ensure there is at least one neuron in the layer.
        assert!(n_neurons > 0);
        let mut random_weights: Vec<f64> = Vec::new();
        // If there is a previous layer, then add random weights to this layer of the neural network.
        if previous_layer > 0 {
            for _ in 1..previous_layer {
                random_weights.push(rand::random());
            }
        }
        // Add neurons to the layer.
        let mut neurons: Vec<neuron::Neuron> = Vec::new();
        for _ in 1..n_neurons {
            neurons.push(neuron::Neuron::new(&random_weights, &lr, activation_function))
        }
        let mut output_cache: Vec<f64> = Vec::new();
        for _ in 1..n_neurons {
            output_cache.push(0.0);
        }
        // Return the layer.
        Layer {
            previous_layer,
            neurons,
            output_cache,
        }
    }
    pub fn outputs(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        if self.previous_layer == 0 {
            self.output_cache = inputs;
        } else {
            self.output_cache.clear();
            for n in 0..(self.neurons.len() - 1) {
                self.output_cache.push(self.neurons[n].output(&inputs));
            }
        }
        self.output_cache
    }
    /// * Calculates the deltas for the last layer of the neural network.
    /// y -> The actual values (that the neural network should have predicted). Used to calculate the loss and then update weights.
    pub fn get_deltas_for_last_layer(&mut self, y: Vec<f64>) {
        for n in 0..(self.neurons.len() - 1) {
            self.neurons[n].delta = (self.neurons[n].activation_function.derivative_function)(self.neurons[n].output_cache) * (y[n] - self.output_cache[n])
        }
    }
    pub fn get_delates_for_hidden_layer(&mut self, next_layer: &mut Layer) {
        for i in 0..(self.neurons.len() - 1) {

            // Calculate next layer.
            let mut next_weights: Vec<f64> = Vec::new();
            let mut next_deltas: Vec<f64> = Vec::new();
            for neuron in next_layer.neurons {
                next_weights.push(neuron.weights[i]);
                next_deltas.push(neuron.delta);
            }
            // Sum weights + deltas
            let sum_of_weights_and_deltas: f64 = dot_product(&next_weights, &next_deltas);
            // n (mutable reference to neuron) is created here
            self.neurons[i].delta = sum_of_weights_and_deltas * (self.neurons[i].activation_function.derivative_function)(self.neurons[i].output_cache) as f64
            // n (mutable reference to neuron) is dropped here
        }
    }
}