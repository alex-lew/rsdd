use super::semiring_traits::*;
use super::logsemiring::LogSemiring;
use std::{fmt::Display, ops};
use serde::{Serialize, Deserialize};

/// A dual number in log-space for automatic differentiation.
///
/// Stores (log_value, d_log_value) where d_log_value = ∂(log_value)/∂θ.
/// This representation is natural for computing gradients of log-probabilities.
///
/// Operations:
/// - Multiplication: (log_a, d_log_a) × (log_b, d_log_b) = (log_a + log_b, d_log_a + d_log_b)
/// - Addition: Uses logsumexp with softmax-weighted gradients
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LogDualNumber(pub f64, pub Vec<f64>);

impl Display for LogDualNumber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {:?})", self.0, self.1)
    }
}

/// Helper function to merge derivative vectors with a binary operation.
/// Handles empty vectors (which signal "adopt size from context") and mismatched sizes.
fn merge_derivs<F>(a: &[f64], b: &[f64], op: F) -> Vec<f64>
where
    F: Fn(f64, f64) -> f64,
{
    if a.is_empty() && b.is_empty() {
        return vec![];
    }
    if a.is_empty() {
        return b.iter().map(|&x| op(0.0, x)).collect();
    }
    if b.is_empty() {
        return a.iter().map(|&x| op(x, 0.0)).collect();
    }

    let max_len = std::cmp::max(a.len(), b.len());
    let mut result = vec![0.0; max_len];
    for i in 0..max_len {
        let a_val = if i < a.len() { a[i] } else { 0.0 };
        let b_val = if i < b.len() { b[i] } else { 0.0 };
        result[i] = op(a_val, b_val);
    }
    result
}

impl ops::Add for LogDualNumber {
    type Output = Self;

    /// Addition in log-space: logsumexp with softmax-weighted gradients.
    ///
    /// z = logsumexp(log_a, log_b)
    /// d_z = softmax_a × d_log_a + softmax_b × d_log_b
    ///
    /// where softmax_a = exp(log_a - z), softmax_b = exp(log_b - z)
    fn add(self, rhs: Self) -> Self::Output {
        // Handle -∞ cases first
        if self.0 == f64::NEG_INFINITY {
            return rhs;
        }
        if rhs.0 == f64::NEG_INFINITY {
            return self;
        }

        let z = LogSemiring::logsumexp(self.0, rhs.0);

        // Compute softmax weights for gradient combination
        let softmax_a = (self.0 - z).exp();
        let softmax_b = (rhs.0 - z).exp();

        // d_z = softmax_a * d_log_a + softmax_b * d_log_b
        let derivs = merge_derivs(&self.1, &rhs.1, |d_log_a, d_log_b| {
            softmax_a * d_log_a + softmax_b * d_log_b
        });

        LogDualNumber(z, derivs)
    }
}

impl ops::Mul for LogDualNumber {
    type Output = Self;

    /// Multiplication in log-space: both value and gradient add.
    ///
    /// log(a × b) = log_a + log_b
    /// d(log(a × b)) = d_log_a + d_log_b
    fn mul(self, rhs: Self) -> Self::Output {
        let derivs = merge_derivs(&self.1, &rhs.1, |d_log_a, d_log_b| d_log_a + d_log_b);
        LogDualNumber(self.0 + rhs.0, derivs)
    }
}

impl ops::Sub for LogDualNumber {
    type Output = Self;

    /// Subtraction in log-space (for Ring trait).
    /// This is tricky and only valid when self > rhs.
    fn sub(self, rhs: Self) -> Self::Output {
        if rhs.0 == f64::NEG_INFINITY {
            return self;
        }
        if self.0 == f64::NEG_INFINITY {
            // log(0 - b) is undefined
            return LogDualNumber(f64::NAN, vec![]);
        }

        // log(a - b) = log(a) + log(1 - exp(log(b) - log(a)))
        let diff = rhs.0 - self.0;
        if diff >= 0.0 {
            // b >= a, result is non-positive (can't represent in log-space)
            return LogDualNumber(f64::NEG_INFINITY, vec![]);
        }

        let exp_diff = diff.exp();
        let log_factor = (1.0 - exp_diff).ln();
        let z = self.0 + log_factor;

        // Gradient computation for log(a - b):
        // d/dθ log(a - b) = (a × d_log_a - b × d_log_b) / (a - b)
        //                 = (exp(log_a) × d_log_a - exp(log_b) × d_log_b) / exp(z)
        //                 = exp(log_a - z) × d_log_a - exp(log_b - z) × d_log_b
        let weight_a = (self.0 - z).exp();
        let weight_b = (rhs.0 - z).exp();

        let derivs = merge_derivs(&self.1, &rhs.1, |d_log_a, d_log_b| {
            weight_a * d_log_a - weight_b * d_log_b
        });

        LogDualNumber(z, derivs)
    }
}

impl Semiring for LogDualNumber {
    fn one() -> Self {
        LogDualNumber(0.0, vec![]) // log(1) = 0, empty gradient adopts size from context
    }

    fn zero() -> Self {
        LogDualNumber(f64::NEG_INFINITY, vec![]) // log(0) = -∞
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mul_adds_both() {
        let a = LogDualNumber(2.0, vec![1.0, 0.0]);
        let b = LogDualNumber(3.0, vec![0.0, 1.0]);
        let result = a * b;
        assert_eq!(result.0, 5.0);
        assert_eq!(result.1, vec![1.0, 1.0]);
    }

    #[test]
    fn test_add_softmax_weights() {
        // If both have equal log-values, softmax weights are 0.5 each
        let a = LogDualNumber(0.0, vec![2.0, 0.0]);
        let b = LogDualNumber(0.0, vec![0.0, 2.0]);
        let result = a + b;
        // logsumexp(0, 0) = log(2)
        assert!((result.0 - 2.0_f64.ln()).abs() < 1e-10);
        // Gradients: 0.5 * 2.0 + 0.5 * 0.0 = 1.0 for first component
        assert!((result.1[0] - 1.0).abs() < 1e-10);
        assert!((result.1[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_add_with_neg_inf() {
        let a = LogDualNumber(1.0, vec![1.0]);
        let b = LogDualNumber::zero();
        let result = a.clone() + b;
        assert_eq!(result.0, a.0);
        assert_eq!(result.1, a.1);
    }

    #[test]
    fn test_identities() {
        let a = LogDualNumber(2.0, vec![1.0, 2.0]);
        let one = LogDualNumber::one();
        let zero = LogDualNumber::zero();

        // a * 1 = a (value is 2.0 + 0.0 = 2.0, gradient adds empty = same)
        let a_times_one = a.clone() * one.clone();
        assert_eq!(a_times_one.0, a.0);
        assert_eq!(a_times_one.1, a.1);

        // a + 0 = a (zero has -inf value, so just returns a)
        let a_plus_zero = a.clone() + zero;
        assert_eq!(a_plus_zero.0, a.0);
        assert_eq!(a_plus_zero.1, a.1);
    }

    #[test]
    fn test_empty_derivs_adopt_size() {
        let a = LogDualNumber(1.0, vec![]);
        let b = LogDualNumber(1.0, vec![1.0, 2.0, 3.0]);

        // Multiplication: empty + vec = vec
        let result = a.clone() * b.clone();
        assert_eq!(result.1.len(), 3);

        // Addition: empty gradient treated as zeros
        let result = a + b;
        // softmax weights are 0.5 each since values are equal
        // 0.5 * 0.0 + 0.5 * 1.0 = 0.5, etc.
        assert!((result.1[0] - 0.5).abs() < 1e-10);
    }
}
