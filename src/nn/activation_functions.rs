pub fn sigmoid(x: f64) -> f64 {
    1 / (1 + (-x).exp())
}

pub fn derivative_sigmoid(x: f64) -> f64 {
    sigmoid(x) * (1 - sigmoid(x))
}

pub struct ActivationFunction {
    function: fn(f64) -> f64,
    derivative_function: fn(f64) -> f64,
}

pub enum ActivationTypes {
    Sigmoid,
}

impl ActivationFunction {
    fn new(function_type: ActivationTypes) -> ActivationFunction {
        match function_type {
            Sigmoid => {
                return ActivationFunction {
                    function: sigmoid,
                    derivative_function: derivative_sigmoid,
                };
            }
        }
    }
}
