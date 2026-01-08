use super::semiring_traits::*;
use std::{fmt::Display, ops};
use serde::{Serialize, Deserialize};

/// A semiring that operates in log-space to avoid numerical underflow.
///
/// Values are stored as log-probabilities:
/// - Zero (additive identity) = -∞ (log(0))
/// - One (multiplicative identity) = 0 (log(1))
/// - Addition = logsumexp (log of sum of exponentials)
/// - Multiplication = addition in log-space
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct LogSemiring(pub f64);

impl LogSemiring {
    /// Compute log(exp(a) + exp(b)) in a numerically stable way.
    /// Uses the identity: logsumexp(a, b) = max(a, b) + log(1 + exp(-|a - b|))
    #[inline]
    pub fn logsumexp(a: f64, b: f64) -> f64 {
        if a == f64::NEG_INFINITY {
            return b;
        }
        if b == f64::NEG_INFINITY {
            return a;
        }
        let max = a.max(b);
        let min = a.min(b);
        max + (1.0 + (min - max).exp()).ln()
    }
}

impl Display for LogSemiring {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl ops::Add<LogSemiring> for LogSemiring {
    type Output = LogSemiring;

    /// Addition in log-space: logsumexp
    /// log(a + b) = logsumexp(log(a), log(b))
    fn add(self, rhs: LogSemiring) -> Self::Output {
        LogSemiring(Self::logsumexp(self.0, rhs.0))
    }
}

impl ops::Mul<LogSemiring> for LogSemiring {
    type Output = LogSemiring;

    /// Multiplication in log-space: addition
    /// log(a * b) = log(a) + log(b)
    fn mul(self, rhs: LogSemiring) -> Self::Output {
        LogSemiring(self.0 + rhs.0)
    }
}

impl ops::Sub<LogSemiring> for LogSemiring {
    type Output = LogSemiring;

    /// Subtraction in log-space: log(a - b) = log(a) + log(1 - exp(log(b) - log(a)))
    /// This is only valid when a > b (i.e., log(a) > log(b))
    fn sub(self, rhs: LogSemiring) -> Self::Output {
        if rhs.0 == f64::NEG_INFINITY {
            return self;
        }
        if self.0 == f64::NEG_INFINITY {
            // log(0 - b) is undefined/negative
            return LogSemiring(f64::NAN);
        }
        // log(a - b) = log(a) + log(1 - exp(log(b) - log(a)))
        let diff = rhs.0 - self.0;
        if diff >= 0.0 {
            // b >= a, so a - b <= 0, which can't be represented in log-space
            return LogSemiring(f64::NEG_INFINITY);
        }
        LogSemiring(self.0 + (1.0 - diff.exp()).ln())
    }
}

impl Semiring for LogSemiring {
    fn one() -> Self {
        LogSemiring(0.0) // log(1) = 0
    }

    fn zero() -> Self {
        LogSemiring(f64::NEG_INFINITY) // log(0) = -∞
    }
}

impl Ring for LogSemiring {}

impl JoinSemilattice for LogSemiring {
    fn join(&self, arg: &Self) -> Self {
        LogSemiring(f64::max(self.0, arg.0))
    }
}

impl BBSemiring for LogSemiring {
    fn choose(&self, arg: &LogSemiring) -> LogSemiring {
        JoinSemilattice::join(self, arg)
    }
}

impl BBRing for LogSemiring {
    fn choose(&self, arg: &LogSemiring) -> LogSemiring {
        JoinSemilattice::join(self, arg)
    }
}

impl MeetSemilattice for LogSemiring {
    fn meet(&self, arg: &Self) -> Self {
        LogSemiring(f64::min(self.0, arg.0))
    }
}

impl Lattice for LogSemiring {}

impl EdgeboundingRing for LogSemiring {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logsumexp_basic() {
        // logsumexp(log(0.5), log(0.5)) = log(1.0) = 0
        let a = LogSemiring(0.5_f64.ln());
        let b = LogSemiring(0.5_f64.ln());
        let result = a + b;
        assert!((result.0 - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_logsumexp_with_neg_inf() {
        let a = LogSemiring(1.0);
        let b = LogSemiring::zero();
        assert_eq!((a + b).0, 1.0);
        assert_eq!((b + a).0, 1.0);
    }

    #[test]
    fn test_mul_is_add() {
        let a = LogSemiring(2.0);
        let b = LogSemiring(3.0);
        assert_eq!((a * b).0, 5.0);
    }

    #[test]
    fn test_identities() {
        let a = LogSemiring(2.0);
        let one = LogSemiring::one();
        let zero = LogSemiring::zero();

        // a * 1 = a
        assert_eq!((a * one).0, a.0);
        // a + 0 = a
        assert_eq!((a + zero).0, a.0);
        // a * 0 = 0
        assert_eq!((a * zero).0, f64::NEG_INFINITY);
    }
}
