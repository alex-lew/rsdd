use super::semiring_traits::*;
use std::{fmt::Display, ops};

#[derive(Clone, Debug, PartialEq, Copy, Serialize, Deserialize)]
pub struct RealSemiringDeriv(pub f64, pub f64);

impl Display for RealSemiringDeriv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.0, self.1)
    }
}

impl ops::Add<RealSemiringDeriv> for RealSemiringDeriv {
    type Output = RealSemiringDeriv;

    fn add(self, rhs: RealSemiringDeriv) -> Self::Output {
        RealSemiringDeriv(self.0 + rhs.0, self.1 + rhs.1)
    }
}

impl ops::Mul<RealSemiringDeriv> for RealSemiringDeriv {
    type Output = RealSemiringDeriv;

    fn mul(self, rhs: RealSemiringDeriv) -> Self::Output {
        RealSemiringDeriv(self.0 * rhs.0, (self.0 * rhs.1) + (self.1 * rhs.0))
    }
}

impl ops::Sub<RealSemiringDeriv> for RealSemiringDeriv {
    type Output = RealSemiringDeriv;

    fn sub(self, rhs: RealSemiringDeriv) -> Self::Output {
        RealSemiringDeriv(self.0 - rhs.0, self.1 - rhs.1)
    }
}

impl Semiring for RealSemiringDeriv {
    fn one() -> Self {
        RealSemiringDeriv(1.0, 0.0)
    }

    fn zero() -> Self {
        RealSemiringDeriv(0.0, 0.0)
    }
}
