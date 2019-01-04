use std::option;
use super::neuron;

pub enum PreviousLayer<Layer> {
    None,
    Some(Layer),
}

pub struct Layer {
    previous_layer: option<Layer>,
    neurons: Vec<Neuron>,
    output_cache: Vec<f64>,
}

impl layer {
    fn new(previous_layer: option<Layer>) -> Layer {}
}