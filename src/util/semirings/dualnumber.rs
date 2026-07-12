// use super::semiring_traits::*;
// use std::{fmt::Display, ops};
// use serde::{Serialize, Deserialize};

// #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
// pub struct DualNumber(pub f64, pub Vec<f64>);

// impl Display for DualNumber {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "({}, {:?})", self.0, self.1)
//     }
// }

// impl ops::Add for DualNumber {
//     type Output = Self;

// /*************  ✨ Windsurf Command ⭐  *************/
//     /// The derivative of the addition of two dual numbers is the element-wise
//     /// sum of the derivatives of the two numbers.
// /*******  fd29ce1e-1a52-4c1b-9db4-dad1dc8a287a  *******/    fn add(self, rhs: Self) -> Self::Output {
//         let mut result_derivs = vec![0.0; self.1.len()];
//         for i in 0..self.1.len() {
//             result_derivs[i] = self.1[i] + rhs.1[i];
//         }
//         Self(self.0 + rhs.0, result_derivs)
//     }
// }

// impl ops::Mul for DualNumber {
//     type Output = Self;

//     fn mul(self, rhs: Self) -> Self::Output {
//         println!("Multiplying: {} * {}", self, rhs);
//         let mut result_derivs = vec![0.0; self.1.len()];
//         for i in 0..self.1.len() {
//             result_derivs[i] = self.0 * rhs.1[i] + self.1[i] * rhs.0;
//         }
//         Self(self.0 * rhs.0, result_derivs)
//     }
// }

// impl ops::Sub for DualNumber {
//     type Output = Self;

//     fn sub(self, rhs: Self) -> Self::Output {
//         let mut result_derivs = vec![0.0; self.1.len()];
//         for i in 0..self.1.len() {
//             result_derivs[i] = self.1[i] - rhs.1[i];
//         }
//         Self(self.0 - rhs.0, result_derivs)
//     }
// }

// impl Semiring for DualNumber {
//     fn one() -> Self {
//         Self(1.0, vec![]) // Default size of 3, but can be any size now
//     }

//     fn zero() -> Self {
//         Self(0.0, vec![]) // Default size of 3, but can be any size now
//     }
// }

use super::semiring_traits::*;
use std::{fmt::Display, ops};
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DualNumber(pub f64, pub Vec<f64>);

impl Display for DualNumber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {:?})", self.0, self.1)
    }
}

impl ops::Add for DualNumber {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        if self.1.is_empty() && !rhs.1.is_empty() {
            // Use rhs's size
            let mut result_derivs = vec![0.0; rhs.1.len()];
            for i in 0..rhs.1.len() {
                result_derivs[i] = rhs.1[i];
            }
            return Self(self.0 + rhs.0, result_derivs);
        } else if !self.1.is_empty() && rhs.1.is_empty() {
            // Use self's size
            let result_derivs = self.1.clone();
            return Self(self.0 + rhs.0, result_derivs);
        } else if self.1.is_empty() && rhs.1.is_empty() {
            // Both empty, return empty
            return Self(self.0 + rhs.0, vec![]);
        }
        
        // Normal case: both vectors have values
        // Ensure they have the same length
        if self.1.len() != rhs.1.len() {
            // Handle mismatched sizes - use the larger one
            let max_size = std::cmp::max(self.1.len(), rhs.1.len());
            let mut result_derivs = vec![0.0; max_size];
            
            // Copy values from self
            for (i, &val) in self.1.iter().enumerate().take(max_size) {
                result_derivs[i] += val;
            }
            
            // Add values from rhs
            for (i, &val) in rhs.1.iter().enumerate().take(max_size) {
                result_derivs[i] += val;
            }
            
            return Self(self.0 + rhs.0, result_derivs);
        }
        
        // Same size vectors
        let mut result_derivs = vec![0.0; self.1.len()];
        for i in 0..self.1.len() {
            result_derivs[i] = self.1[i] + rhs.1[i];
        }
        Self(self.0 + rhs.0, result_derivs)
    }
}

impl ops::Mul for DualNumber {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.1.is_empty() && !rhs.1.is_empty() {
            // Use rhs's size but apply multiplication formula
            let mut result_derivs = vec![0.0; rhs.1.len()];
            for i in 0..rhs.1.len() {
                result_derivs[i] = self.0 * rhs.1[i];  // Since self's derivatives are empty
            }
            return Self(self.0 * rhs.0, result_derivs);
        } else if !self.1.is_empty() && rhs.1.is_empty() {
            // Use self's size but apply multiplication formula
            let mut result_derivs = vec![0.0; self.1.len()];
            for i in 0..self.1.len() {
                result_derivs[i] = self.1[i] * rhs.0;  // Since rhs's derivatives are empty
            }
            return Self(self.0 * rhs.0, result_derivs);
        } else if self.1.is_empty() && rhs.1.is_empty() {
            // Both empty, return empty
            return Self(self.0 * rhs.0, vec![]);
        }
        
        // Normal case: both vectors have values
        // Ensure they have the same length
        if self.1.len() != rhs.1.len() {
            // Handle mismatched sizes - use the larger one
            let max_size = std::cmp::max(self.1.len(), rhs.1.len());
            let mut result_derivs = vec![0.0; max_size];
            
            // Apply multiplication formula with bounds checking
            for i in 0..max_size {
                let self_deriv = if i < self.1.len() { self.1[i] } else { 0.0 };
                let rhs_deriv = if i < rhs.1.len() { rhs.1[i] } else { 0.0 };
                
                result_derivs[i] = self.0 * rhs_deriv + self_deriv * rhs.0;
            }
            
            return Self(self.0 * rhs.0, result_derivs);
        }
        
        // Same size vectors
        let mut result_derivs = vec![0.0; self.1.len()];
        for i in 0..self.1.len() {
            result_derivs[i] = self.0 * rhs.1[i] + self.1[i] * rhs.0;
        }
        Self(self.0 * rhs.0, result_derivs)
    }
}

impl ops::Sub for DualNumber {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        if self.1.is_empty() && !rhs.1.is_empty() {
            // Use rhs's size with negation
            let mut result_derivs = vec![0.0; rhs.1.len()];
            for i in 0..rhs.1.len() {
                result_derivs[i] = -rhs.1[i];  // Negate rhs's derivatives
            }
            return Self(self.0 - rhs.0, result_derivs);
        } else if !self.1.is_empty() && rhs.1.is_empty() {
            // Use self's size
            let result_derivs = self.1.clone();
            return Self(self.0 - rhs.0, result_derivs);
        } else if self.1.is_empty() && rhs.1.is_empty() {
            // Both empty, return empty
            return Self(self.0 - rhs.0, vec![]);
        }
        
        // Normal case: both vectors have values
        // Ensure they have the same length
        if self.1.len() != rhs.1.len() {
            // Handle mismatched sizes - use the larger one
            let max_size = std::cmp::max(self.1.len(), rhs.1.len());
            let mut result_derivs = vec![0.0; max_size];
            
            // Copy values from self
            for (i, &val) in self.1.iter().enumerate().take(max_size) {
                result_derivs[i] += val;
            }
            
            // Subtract values from rhs
            for (i, &val) in rhs.1.iter().enumerate().take(max_size) {
                result_derivs[i] -= val;
            }
            
            return Self(self.0 - rhs.0, result_derivs);
        }
        
        // Same size vectors
        let mut result_derivs = vec![0.0; self.1.len()];
        for i in 0..self.1.len() {
            result_derivs[i] = self.1[i] - rhs.1[i];
        }
        Self(self.0 - rhs.0, result_derivs)
    }
}

impl Semiring for DualNumber {
    fn one() -> Self {
        Self(1.0, vec![])  // Empty vector signals "adopt size from context"
    }

    fn zero() -> Self {
        Self(0.0, vec![])  // Empty vector signals "adopt size from context"
    }
}