use super::semiring_traits::*;
use std::{fmt::Display, ops};
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, PartialEq, Copy, Serialize, Deserialize)]
pub struct DualNumber<const N: usize>(pub f64, pub [f64; N]);

impl<const N: usize> Display for DualNumber<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {:?})", self.0, self.1)
    }
}

impl<const N: usize> ops::Add for DualNumber<N> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut result_derivs = [0.0; N];
        for i in 0..N {
            result_derivs[i] = self.1[i] + rhs.1[i];
        }
        Self(self.0 + rhs.0, result_derivs)
    }
}

impl<const N: usize> ops::Mul for DualNumber<N> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut result_derivs = [0.0; N];
        for i in 0..N {
            result_derivs[i] = self.0 * rhs.1[i] + self.1[i] * rhs.0;
        }
        Self(self.0 * rhs.0, result_derivs)
    }
}

impl<const N: usize> ops::Sub for DualNumber<N> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut result_derivs = [0.0; N];
        for i in 0..N {
            result_derivs[i] = self.1[i] - rhs.1[i];
        }
        Self(self.0 - rhs.0, result_derivs)
    }
}

impl<const N: usize> Semiring for DualNumber<N> {
    fn one() -> Self {
        Self(1.0, [0.0; N])
    }

    fn zero() -> Self {
        Self(0.0, [0.0; N])
    }
}