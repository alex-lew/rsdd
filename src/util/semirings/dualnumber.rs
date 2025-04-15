use super::semiring_traits::*;
use std::{fmt::Display, ops};
use serde::{Serialize, Deserialize};

pub const NPARTIALS: usize = 3;

#[derive(Clone, Debug, PartialEq, Copy, Serialize, Deserialize)]
pub struct DualNumber(pub f64, pub [f64; NPARTIALS]);

impl Display for DualNumber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {:?})", self.0, self.1)
    }
}

impl ops::Add for DualNumber {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut result_derivs = [0.0; NPARTIALS];
        for i in 0..NPARTIALS {
            result_derivs[i] = self.1[i] + rhs.1[i];
        }
        Self(self.0 + rhs.0, result_derivs)
    }
}

impl ops::Mul for DualNumber {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut result_derivs = [0.0; NPARTIALS];
        for i in 0..NPARTIALS {
            result_derivs[i] = self.0 * rhs.1[i] + self.1[i] * rhs.0;
        }
        Self(self.0 * rhs.0, result_derivs)
    }
}

impl ops::Sub for DualNumber {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut result_derivs = [0.0; NPARTIALS];
        for i in 0..NPARTIALS {
            result_derivs[i] = self.1[i] - rhs.1[i];
        }
        Self(self.0 - rhs.0, result_derivs)
    }
}

impl Semiring for DualNumber {
    fn one() -> Self {
        Self(1.0, [0.0; NPARTIALS])
    }

    fn zero() -> Self {
        Self(0.0, [0.0; NPARTIALS])
    }
}