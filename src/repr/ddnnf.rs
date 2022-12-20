//! Implementing of a generic decision decomposable deterministic negation normal form
//! (d-DNNF) pointer type
//! 
//! A decision-DNNF is a DNNF where every or-node is a *decision node*, i.e. the high-path
//! for each disjunction is uniquely given by the configuration of some set of decision
//! variables
use core::fmt::Debug;
use std::collections::HashMap;
use num::Num;

use super::{var_label::{VarLabel, VarSet}, wmc::WmcParams};

/// A base d-DNNF type
pub enum DDNNF<T> {
    /// A tuple (left result, right result, decision set)
    /// The decision set is the set of variables that are decided by this disjunction
    Or(T, T, VarSet),
    And(T, T),
    Lit(VarLabel, bool),
    True,
    False,
}

pub trait DDNNFPtr {
    /// performs a memoized bottom-up pass with aggregating function `f` calls
    fn fold<T: Clone + Copy + Debug, F: Fn(DDNNF<T>) -> T>(&self, f: F) -> T;


    /// Weighted-model count
    fn wmc<T: Num + Clone + Debug + Copy>(&self, params: &WmcParams<T>) -> T {
        self.fold(|ddnnf| {
            use DDNNF::*;
            match ddnnf {
                Or(l, r, _) => l + r,
                And(l, r) => l * r,
                True => params.one,
                False => params.zero,
                Lit(lbl, polarity) => {
                    let (low_w, high_w) = params.get_var_weight(lbl);
                    if polarity {
                        *high_w
                    } else {
                        *low_w
                    }
                }
            }
        })
    }

    fn eval(&self, assgn: &HashMap<VarLabel, bool>) -> bool {
        todo!()
    }

    /// count the number of nodes in this representation
    fn count_nodes(&self) -> usize;
}
