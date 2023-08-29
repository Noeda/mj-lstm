#[derive(Clone, Debug, Copy, PartialEq, PartialOrd)]
pub struct AdamWConfiguration {
    pub(crate) beta1: f64,
    pub(crate) beta2: f64,
    pub(crate) epsilon: f64,
    pub(crate) learning_rate: f64,
    pub(crate) weight_decay: f64,
    pub(crate) gradient_clip: f64,
}

impl AdamWConfiguration {
    pub fn new() -> Self {
        AdamWConfiguration {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            learning_rate: 0.001,
            weight_decay: 0.0,
            gradient_clip: std::f64::INFINITY,
        }
    }

    pub fn gradient_clip(self, gradient_clip: f64) -> Self {
        Self {
            gradient_clip,
            ..self
        }
    }

    pub fn learning_rate(self, learning_rate: f64) -> Self {
        Self {
            learning_rate,
            ..self
        }
    }

    pub fn weight_decay(self, weight_decay: f64) -> Self {
        Self {
            weight_decay,
            ..self
        }
    }

    pub fn beta1(self, beta1: f64) -> Self {
        Self { beta1, ..self }
    }

    pub fn beta2(self, beta2: f64) -> Self {
        Self { beta2, ..self }
    }

    pub fn epsilon(self, epsilon: f64) -> Self {
        Self { epsilon, ..self }
    }
}

impl Default for AdamWConfiguration {
    fn default() -> Self {
        Self::new()
    }
}
