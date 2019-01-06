pub mod activations {
    pub fn sigmoid(x: &f64) -> f64 {
        1 as f64 / (1 as f64 + (-x).exp())
    }

    pub fn derivative_sigmoid(x: &f64) -> f64 {
        sigmoid(x) * (1 as f64 - sigmoid(x))
    }

    pub enum ActivationTypes {
        Sigmoid
    }

    // Struct to hold the activation function.
    #[derive(Debug, Copy, Clone)]
    pub struct ActivationFunction {
        pub function: fn(&f64) -> f64,
        pub derivative_function: fn(&f64) -> f64,
    }

    impl ActivationFunction {
        fn new(function_type: ActivationTypes) -> ActivationFunction {
            match function_type {
                ActivationTypes::Sigmoid => {
                    ActivationFunction {
                        function: sigmoid,
                        derivative_function: derivative_sigmoid,
                    }
                }
            }
        }
    }
}