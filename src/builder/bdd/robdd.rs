use ordered_float::OrderedFloat;
use rand::{rngs::ThreadRng, Rng};

use crate::{
    backing_store::{BackedRobinhoodTable, UniqueTable},
    builder::{
        bdd::{BddBuilder, BddBuilderStats},
        cache::{Ite, IteTable},
        BottomUpBuilder,
    },
    repr::{BddNode, BddPtr, DDNNFPtr, PartialModel, VarLabel, VarOrder, WmcParams},
    util::semirings::RealSemiring,
};
use std::{cell::RefCell, time::{Duration, Instant}};

pub struct RobddBuilder<'a, T: IteTable<'a, BddPtr<'a>> + Default> {
    compute_table: RefCell<BackedRobinhoodTable<'a, BddNode<'a>>>,
    apply_table: RefCell<T>,
    stats: RefCell<BddBuilderStats>,
    order: RefCell<VarOrder>,
    time_limit: Option<(Instant, Duration)>,
}

type SampleCache = (Option<f64>, Option<f64>);

impl<'a, T: IteTable<'a, BddPtr<'a>> + Default> BddBuilder<'a> for RobddBuilder<'a, T> {
    fn less_than(&self, a: VarLabel, b: VarLabel) -> bool {
        self.order.borrow().lt(a, b)
    }

    fn has_variable(&self, bdd: BddPtr<'a>, var: VarLabel) -> bool {
        match bdd {
            BddPtr::PtrTrue | BddPtr::PtrFalse => false,
            BddPtr::Compl(node) | BddPtr::Reg(node) => {
                if node.var == var {
                    true
                } else if self.less_than(var, node.var) {
                    false // If var should come before node.var in the order, it won't appear below
                } else {
                    self.has_variable(node.low, var) || self.has_variable(node.high, var)
                }
            }
        }
    }

    /// Normalizes and fetches a node from the store
    fn get_or_insert(&'a self, bdd: BddNode<'a>) -> BddPtr<'a> {
        unsafe {
            // TODO: Make this safe if possible
            let tbl = &mut *self.compute_table.as_ptr();
            if bdd.high.is_neg() || bdd.high.is_false() {
                let bdd: BddNode<'a> = BddNode::new(bdd.var, bdd.low.neg(), bdd.high.neg());
                let r: &'a BddNode<'a> = tbl.get_or_insert(bdd);
                BddPtr::Compl(r)
            } else {
                let bdd = BddNode::new(bdd.var, bdd.low, bdd.high);
                BddPtr::Reg(tbl.get_or_insert(bdd))
            }
        }
    }

    fn ite_helper(&'a self, f: BddPtr<'a>, g: BddPtr<'a>, h: BddPtr<'a>) -> BddPtr<'a> {
        if self.check_time_limit() {
            return BddPtr::PtrFalse; // doesn't matter what we return here, our callee is responsible for checking the time limit
        }

        self.stats.borrow_mut().num_recursive_calls += 1;
        let o = |a: BddPtr, b: BddPtr| match (a, b) {
            (BddPtr::PtrTrue, _) | (BddPtr::PtrFalse, _) => true,
            (_, BddPtr::PtrTrue) | (_, BddPtr::PtrFalse) => false,
            (
                BddPtr::Reg(node_a) | BddPtr::Compl(node_a),
                BddPtr::Reg(node_b) | BddPtr::Compl(node_b),
            ) => self.order.borrow().lt(node_a.var, node_b.var),
        };

        let ite = Ite::new(o, f, g, h);

        if let Ite::IteConst(f) = ite {
            return f;
        }

        let hash = self.apply_table.borrow().hash(&ite);
        if let Some(v) = self.apply_table.borrow().get(ite, hash) {
            return v;
        }

        // ok the work!
        // find the first essential variable for f, g, or h
        let lbl = self.order.borrow().first_essential(&f, &g, &h);
        let fx = self.condition_essential(f, lbl, true);
        let gx = self.condition_essential(g, lbl, true);
        let hx = self.condition_essential(h, lbl, true);
        let fxn = self.condition_essential(f, lbl, false);
        let gxn = self.condition_essential(g, lbl, false);
        let hxn = self.condition_essential(h, lbl, false);
        let t = self.ite(fx, gx, hx);
        let f = self.ite(fxn, gxn, hxn);

        if t == f {
            self.apply_table.borrow_mut().insert(ite, t, hash);
            return t;
        };

        if self.check_time_limit() {
            // to avoid us caching this in apply_table
            return BddPtr::PtrFalse;
        }

        // now we have a new BDD
        let node = BddNode::new(lbl, f, t);
        let r = self.get_or_insert(node);
        self.apply_table.borrow_mut().insert(ite, r, hash);
        r
    }

    fn cond_helper(&'a self, bdd: BddPtr<'a>, lbl: VarLabel, value: bool) -> BddPtr<'a> {
        self.cond_with_alloc(bdd, lbl, value, &mut Vec::new())
    }
}

impl<'a, T: IteTable<'a, BddPtr<'a>> + Default> RobddBuilder<'a, T> {
    /// Creates a new variable manager with the specified order
    pub fn new(order: VarOrder, time_limit: Option<(Instant, Duration)>) -> RobddBuilder<'a, T> {
        RobddBuilder {
            compute_table: RefCell::new(BackedRobinhoodTable::new()),
            order: RefCell::new(order),
            apply_table: RefCell::new(T::default()),
            stats: RefCell::new(BddBuilderStats::new()),
            time_limit,
        }
    }

    /// Make a BDD manager with a default variable ordering
    pub fn new_with_linear_order(num_vars: usize) -> RobddBuilder<'a, T> {
        let default_order = VarOrder::linear_order(num_vars);
        RobddBuilder::new(default_order, None)
    }

    pub fn start_time_limit(&mut self, time_limit: Duration) {
        self.time_limit = Some((Instant::now(), time_limit));
    }
    pub fn stop_time_limit(&mut self) {
        self.time_limit = None;
    }

    #[inline(always)]
    pub fn check_time_limit(&self) -> bool {
        if let Some((start_time, time_limit)) = self.time_limit {
            return start_time.elapsed() > time_limit;
        }
        false
    }

    /// Returns the number of variables in the manager
    #[inline]
    pub fn num_vars(&self) -> usize {
        self.order.borrow().num_vars()
    }

    /// Generate a new variable label which was not in the original order. Places the
    /// new variable label at the end of the current order. Returns the newly
    /// generated label.
    #[inline]
    pub fn new_label(&self) -> VarLabel {
        self.order.borrow_mut().new_last()
    }

    /// Generate a new variable label and insert it at the specified position in the current order.
    /// Returns the newly generated label.
    ///
    /// # Arguments
    ///
    /// * `position` - The position at which to insert the new variable in the order.
    ///
    /// # Panics
    ///
    /// Panics if the position is out of bounds for the current order.
    #[inline]
    pub fn new_label_at_position(&self, position: usize) -> VarLabel {
        self.order.borrow_mut().insert_var_at_position(position)
    }

    /// Generate a new pointer and insert it at the specified position in the current order.
    /// Returns the newly generated label and the corresponding BDD pointer.
    ///
    /// # Arguments
    ///
    /// * `position` - The position at which to insert the new variable in the order.
    /// * `polarity` - The polarity of the new variable (true for positive, false for negative).
    ///
    /// # Panics
    ///
    /// Panics if the position is out of bounds for the current order.
    #[inline]
    pub fn new_var_at_position(
        &'a self,
        position: usize,
        polarity: bool,
    ) -> (VarLabel, BddPtr<'a>) {
        let label = self.new_label_at_position(position);
        let ptr = self.var(label, polarity);
        (label, ptr)
    }

    /// Generate a new pointer which was not in the original order. Uses
    /// `new_label` to produce a new label at the end of the current order, then
    /// uses `var` to create a pointer in the manager. Returns the output of both.
    #[inline]
    pub fn new_var(&'a self, polarity: bool) -> (VarLabel, BddPtr<'a>) {
        let label = self.new_label();
        let ptr = self.var(label, polarity);
        (label, ptr)
    }

    /// Use `new_var` to create a new positive pointer.
    #[inline]
    pub fn new_pos(&'a self) -> (VarLabel, BddPtr<'a>) {
        self.new_var(true)
    }

    /// Use `new_var` to create a new negative pointer.
    #[inline]
    pub fn new_neg(&'a self) -> (VarLabel, BddPtr<'a>) {
        self.new_var(false)
    }

    pub fn weighted_sample(
        &'a self,
        ptr: BddPtr<'a>,
        wmc: &WmcParams<RealSemiring>,
    ) -> (BddPtr<'a>, f64) {
        let mut rng = rand::thread_rng();

        fn bottomup_pass_h(ptr: BddPtr, wmc: &WmcParams<RealSemiring>) -> f64 {
            match ptr {
                BddPtr::PtrTrue => 1.0,
                BddPtr::PtrFalse => 0.0,
                BddPtr::Compl(node) | BddPtr::Reg(node) => {
                    // inside the cache, store a (compl, non_compl) pair corresponding to the
                    // complemented and uncomplemented pass over this node

                    // helper performs actual fold-and-cache work
                    let bottomup_helper = |cached| {
                        let (l, h) = if ptr.is_neg() {
                            (ptr.low_raw().neg(), ptr.high_raw().neg())
                        } else {
                            (ptr.low_raw(), ptr.high_raw())
                        };

                        let low_v = bottomup_pass_h(l, wmc);
                        let high_v = bottomup_pass_h(h, wmc);
                        let top = node.var;

                        let and_low = wmc.var_weight(top).0 .0 * low_v;
                        let and_high = wmc.var_weight(top).1 .0 * high_v;

                        let or_v = and_low + and_high;

                        // cache and return or_v
                        if ptr.is_neg() {
                            ptr.set_scratch::<SampleCache>((Some(or_v), cached));
                        } else {
                            ptr.set_scratch::<SampleCache>((cached, Some(or_v)));
                        }
                        or_v
                    };

                    match ptr.scratch::<SampleCache>() {
                        // first, check if cached; explicit arms here for clarity
                        Some((Some(l), Some(h))) => {
                            if ptr.is_neg() {
                                l
                            } else {
                                h
                            }
                        }
                        Some((Some(v), None)) if ptr.is_neg() => v,
                        Some((None, Some(v))) if !ptr.is_neg() => v,
                        // no cached value found, compute it
                        Some((None, cached)) | Some((cached, None)) => bottomup_helper(cached),
                        None => bottomup_helper(None),
                    }
                }
            }
        }

        fn sample_path<'b, T: IteTable<'b, BddPtr<'b>> + Default>(
            builder: &'b RobddBuilder<'b, T>,
            ptr: BddPtr<'b>,
            wmc: &WmcParams<RealSemiring>,
            rng: &mut ThreadRng,
        ) -> (BddPtr<'b>, f64) {
            match ptr {
                BddPtr::PtrTrue => (ptr, 1.0),
                BddPtr::PtrFalse => panic!("sample_path called on false!"),
                BddPtr::Compl(node) | BddPtr::Reg(node) => {
                    let (l, h) = if ptr.is_neg() {
                        (ptr.low_raw().neg(), ptr.high_raw().neg())
                    } else {
                        (ptr.low_raw(), ptr.high_raw())
                    };

                    let low_v = bottomup_pass_h(l, wmc);
                    let high_v = bottomup_pass_h(h, wmc);
                    let top = node.var;

                    let and_low = wmc.var_weight(top).0 .0 * low_v;
                    let and_high = wmc.var_weight(top).1 .0 * high_v;

                    // Choose between low and high based on and_low and and_high
                    // Generate a random float between 0 and 1, and then look at
                    // whether it is less than and_low / (and_low + and_high).
                    let total_weight = and_low + and_high;
                    let rand_val = rng.gen_range(0.0..total_weight);
                    if rand_val < and_low {
                        let (low_child, low_child_probability) = sample_path(builder, l, wmc, rng);
                        let new_node = BddNode::new(node.var, low_child, BddPtr::PtrFalse);
                        return (
                            builder.get_or_insert(new_node),
                            low_child_probability * and_low / total_weight,
                        );
                    } else {
                        let (high_child, high_child_probability) =
                            sample_path(builder, h, wmc, rng);
                        let new_node = BddNode::new(node.var, BddPtr::PtrFalse, high_child);
                        return (
                            builder.get_or_insert(new_node),
                            high_child_probability * and_high / total_weight,
                        );
                    }
                }
            }
        }

        // let r = bottomup_pass_h(ptr, wmc);
        let (sample, sample_probability) = sample_path(self, ptr, wmc, &mut rng);
        ptr.clear_scratch();
        (sample, sample_probability)
    }

    /// Compute the top K accepting paths through the BDD and return a new BDD containing only those paths
    pub fn top_k_paths(
        &'a self,
        ptr: BddPtr<'a>,
        k: usize,
        wmc: &WmcParams<RealSemiring>,
    ) -> BddPtr<'a> {
        #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
        struct Path {
            weight: OrderedFloat<f64>,
            decisions: Vec<(VarLabel, bool)>,
        }

        type TopKCache = (Option<Vec<Path>>, Option<Vec<Path>>);

        // Bottom-up pass to compute top K paths
        fn bottom_up_top_k<'b, T: IteTable<'b, BddPtr<'b>> + Default>(
            builder: &'b RobddBuilder<'b, T>,
            ptr: BddPtr<'b>,
            k: usize,
            wmc: &WmcParams<RealSemiring>,
        ) -> Vec<Path> {
            match ptr {
                BddPtr::PtrTrue => vec![Path {
                    weight: OrderedFloat(1.0),
                    decisions: vec![],
                }],
                BddPtr::PtrFalse => vec![],
                BddPtr::Compl(node) | BddPtr::Reg(node) => {
                    let bottomup_helper = |cached: Option<Vec<Path>>| {
                        let (l, h) = if ptr.is_neg() {
                            (ptr.low_raw().neg(), ptr.high_raw().neg())
                        } else {
                            (ptr.low_raw(), ptr.high_raw())
                        };

                        let low_paths = bottom_up_top_k(builder, l, k, wmc);
                        let high_paths = bottom_up_top_k(builder, h, k, wmc);

                        let low_weight = wmc.var_weight(node.var).0 .0;
                        let high_weight = wmc.var_weight(node.var).1 .0;

                        let mut true_paths = Vec::new();

                        true_paths.extend(low_paths.into_iter().map(|mut p| {
                            p.weight *= OrderedFloat(low_weight);
                            p.decisions.insert(0, (node.var, false));
                            p
                        }));

                        true_paths.extend(high_paths.into_iter().map(|mut p| {
                            p.weight *= OrderedFloat(high_weight);
                            p.decisions.insert(0, (node.var, true));
                            p
                        }));

                        true_paths.sort_by(|a, b| b.weight.cmp(&a.weight));
                        true_paths.truncate(k);

                        // println!("Top-k paths for {:?}: {:?}", node.var, true_paths);

                        if ptr.is_neg() {
                            ptr.set_scratch::<TopKCache>((Some(true_paths.clone()), cached));
                        } else {
                            ptr.set_scratch::<TopKCache>((cached, Some(true_paths.clone())));
                        }
                        true_paths
                    };

                    match ptr.scratch::<TopKCache>() {
                        Some((Some(l), Some(h))) => {
                            if ptr.is_neg() {
                                l
                            } else {
                                h
                            }
                        }
                        Some((Some(v), None)) if ptr.is_neg() => v,
                        Some((None, Some(v))) if !ptr.is_neg() => v,
                        Some((None, cached)) | Some((cached, None)) => bottomup_helper(cached),
                        None => bottomup_helper(None),
                    }
                }
            }
        }

        // Top-down pass to construct new BDD with top K paths
        fn construct_top_k_bdd<'b, T: IteTable<'b, BddPtr<'b>> + Default>(
            builder: &'b RobddBuilder<'b, T>,
            paths: &[Path],
            order: &VarOrder,
        ) -> BddPtr<'b> {
            if paths.is_empty() {
                return BddPtr::PtrFalse;
            }

            if paths.iter().all(|p| p.decisions.is_empty()) {
                return BddPtr::PtrTrue;
            }

            // Find the next variable to consider
            let next_var = paths
                .iter()
                .flat_map(|path| path.decisions.first())
                .min_by_key(|&&(var, _)| order.get(var))
                .map(|&(var, _)| var)
                .unwrap();

            let (low_paths, high_paths): (Vec<_>, Vec<_>) = paths.iter().partition(|path| {
                path.decisions
                    .first()
                    .map_or(true, |&(v, d)| v != next_var || !d)
            });

            let low_paths: Vec<_> = low_paths
                .into_iter()
                .map(|p| {
                    let mut new_p = p.clone();
                    if !new_p.decisions.is_empty() && new_p.decisions[0].0 == next_var {
                        new_p.decisions.remove(0);
                    }
                    new_p
                })
                .collect();

            let high_paths: Vec<_> = high_paths
                .into_iter()
                .map(|p| {
                    let mut new_p = p.clone();
                    new_p.decisions.remove(0);
                    new_p
                })
                .collect();

            let low = construct_top_k_bdd(builder, &low_paths, order);
            let high = construct_top_k_bdd(builder, &high_paths, order);

            if low == high {
                low
            } else {
                let new_node = BddNode::new(next_var, low, high);
                builder.get_or_insert(new_node)
            }
        }

        let top_k_paths = bottom_up_top_k(self, ptr, k, wmc);
        let result: BddPtr<'a> = construct_top_k_bdd(self, &top_k_paths, self.order());
        ptr.clear_scratch();
        result
    }

    /// Get the current variable order
    #[inline]
    pub fn order(&self) -> &VarOrder {
        // TODO fix this, it doesn't need to be unsafe
        unsafe { &*self.order.as_ptr() }
    }

    // condition a BDD *only* if the top variable is `v`; used in `ite`
    fn condition_essential(&'a self, f: BddPtr<'a>, lbl: VarLabel, v: bool) -> BddPtr<'a> {
        match f {
            BddPtr::PtrTrue | BddPtr::PtrFalse => f,
            BddPtr::Reg(node) | BddPtr::Compl(node) => {
                if node.var != lbl {
                    return f;
                }
                let r = if v { f.high_raw() } else { f.low_raw() };
                if f.is_neg() {
                    r.neg()
                } else {
                    r
                }
            }
        }
    }

    fn cond_with_alloc(
        &'a self,
        bdd: BddPtr<'a>,
        lbl: VarLabel,
        value: bool,
        alloc: &mut Vec<BddPtr<'a>>,
    ) -> BddPtr<'a> {
        self.stats.borrow_mut().num_recursive_calls += 1;
        match bdd {
            BddPtr::PtrTrue | BddPtr::PtrFalse => bdd,
            BddPtr::Reg(node) | BddPtr::Compl(node) => {
                if self.order.borrow().lt(lbl, node.var) {
                    // we passed the variable in the order, we will never find it
                    return bdd;
                }

                if node.var == lbl {
                    let r = if value { bdd.high_raw() } else { bdd.low_raw() };
                    return if bdd.is_neg() { r.neg() } else { r };
                }

                // check cache
                match bdd.scratch::<usize>() {
                    None => (),
                    Some(v) => {
                        return if bdd.is_neg() {
                            alloc[v].neg()
                        } else {
                            alloc[v]
                        }
                    }
                };

                // recurse on the children
                let l = self.cond_with_alloc(bdd.low_raw(), lbl, value, alloc);
                let h = self.cond_with_alloc(bdd.high_raw(), lbl, value, alloc);

                if l == h {
                    // reduce the BDD -- two children identical
                    if bdd.is_neg() {
                        return l.neg();
                    } else {
                        return l;
                    };
                };
                let res = if l != bdd.low_raw() || h != bdd.high_raw() {
                    // cache and return the new BDD
                    let new_bdd = BddNode::new(node.var, l, h);
                    let r = self.get_or_insert(new_bdd);
                    if bdd.is_neg() {
                        r.neg()
                    } else {
                        r
                    }
                } else {
                    // nothing changed
                    bdd
                };

                let idx = if bdd.is_neg() {
                    alloc.push(res.neg());
                    alloc.len() - 1
                } else {
                    alloc.push(res);
                    alloc.len() - 1
                };
                bdd.set_scratch(idx);
                res
            }
        }
    }

    fn cond_model_h(&'a self, bdd: BddPtr<'a>, m: &PartialModel) -> BddPtr<'a> {
        // TODO: optimize this
        let mut bdd = bdd;
        for m in m.assignment_iter() {
            bdd = self.condition(bdd, m.label(), m.polarity());
        }
        bdd
    }

    /// Compute the Boolean function `f | var = value` for every set value in
    /// the partial model `m`
    ///
    /// Pre-condition: scratch cleared
    pub fn condition_model(&'a self, bdd: BddPtr<'a>, m: &PartialModel) -> BddPtr<'a> {
        debug_assert!(bdd.is_scratch_cleared());
        let r = self.cond_model_h(bdd, m);
        bdd.clear_scratch();
        r
    }

    /// Prints the total number of recursive calls executed so far by the RobddBuilder
    /// This is a stable way to track performance
    pub fn num_recursive_calls(&self) -> usize {
        self.stats.borrow().num_recursive_calls
    }

    fn smooth_helper(&'a self, bdd: BddPtr<'a>, current: usize, total: usize) -> BddPtr<'a> {
        debug_assert!(current <= total);
        if current >= total {
            return bdd;
        }

        match bdd {
            BddPtr::Reg(node) => {
                let smoothed_node = BddNode::new(
                    node.var,
                    self.smooth_helper(node.low, current + 1, total),
                    self.smooth_helper(node.high, current + 1, total),
                );
                self.get_or_insert(smoothed_node)
            }
            BddPtr::Compl(node) => self.smooth_helper(BddPtr::Reg(node), current, total).neg(),
            BddPtr::PtrTrue | BddPtr::PtrFalse => {
                let var = self.order.borrow().var_at_level(current);
                let smoothed_node = BddNode::new(
                    var,
                    self.smooth_helper(bdd, current + 1, total),
                    self.smooth_helper(bdd, current + 1, total),
                );
                self.get_or_insert(smoothed_node)
            }
        }
    }

    /// Return a smoothed version of the input BDD. Requires:
    /// - BDD is an ROBDD, i.e. each variable only appears once per path
    /// - variable ordering respects the builder's order
    pub fn smooth(&'a self, bdd: BddPtr<'a>, num_vars: usize) -> BddPtr<'a> {
        // TODO: this num_vars should be tied to the specific BDD, not the manager
        self.smooth_helper(bdd, 0, num_vars)
    }

    pub fn stats(&'a self) -> BddBuilderStats {
        BddBuilderStats {
            num_recursive_calls: self.stats.borrow().num_recursive_calls,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::builder::BottomUpBuilder;
    use crate::repr::WmcParams;
    use crate::util::semirings::{FiniteField, RealSemiring};
    use crate::{builder::cache::AllIteTable, repr::DDNNFPtr};

    use crate::{
        builder::bdd::robdd::RobddBuilder,
        repr::{BddPtr, Cnf, VarLabel},
    };

    // check that (a \/ b) /\ a === a
    #[test]
    fn simple_equality() {
        let builder = RobddBuilder::<AllIteTable<BddPtr>>::new_with_linear_order(3);
        let v1 = builder.var(VarLabel::new(0), true);
        let v2 = builder.var(VarLabel::new(1), true);
        let r1 = builder.or(v1, v2);
        let r2 = builder.and(r1, v1);
        assert!(
            builder.eq(v1, r2),
            "Not eq:\n {}\n{}",
            v1.to_string_debug(),
            r2.to_string_debug()
        );
    }

    #[test]
    fn simple_ite1() {
        let builder = RobddBuilder::<AllIteTable<BddPtr>>::new_with_linear_order(3);
        let v1 = builder.var(VarLabel::new(0), true);
        let v2 = builder.var(VarLabel::new(1), true);
        let r1 = builder.or(v1, v2);
        let r2 = builder.ite(r1, v1, BddPtr::false_ptr());
        assert!(
            builder.eq(v1, r2),
            "Not eq:\n {}\n{}",
            v1.to_string_debug(),
            r2.to_string_debug()
        );
    }

    #[test]
    fn test_newvar() {
        let builder = RobddBuilder::<AllIteTable<BddPtr>>::new_with_linear_order(0);
        let l1 = builder.new_label();
        let l2 = builder.new_label();
        let v1 = builder.var(l1, true);
        let v2 = builder.var(l2, true);
        let r1 = builder.or(v1, v2);
        let r2 = builder.and(r1, v1);
        assert!(
            builder.eq(v1, r2),
            "Not eq:\n {}\n{}",
            v1.to_string_debug(),
            r2.to_string_debug()
        );
    }

    #[test]
    fn test_wmc() {
        let builder = RobddBuilder::<AllIteTable<BddPtr>>::new_with_linear_order(2);
        let v1 = builder.var(VarLabel::new(0), true);
        let v2 = builder.var(VarLabel::new(1), true);
        let r1 = builder.or(v1, v2);
        let weights = HashMap::from_iter([
            (VarLabel::new(0), (RealSemiring(0.2), RealSemiring(0.8))),
            (VarLabel::new(1), (RealSemiring(0.1), RealSemiring(0.9))),
        ]);
        let params = WmcParams::new(weights);
        let wmc = r1.unsmoothed_wmc(&params);
        assert!((wmc.0 - (1.0 - 0.2 * 0.1)).abs() < 0.000001);
    }

    #[test]
    fn test_condition() {
        let builder = RobddBuilder::<AllIteTable<BddPtr>>::new_with_linear_order(3);
        let v1 = builder.var(VarLabel::new(0), true);
        let v2 = builder.var(VarLabel::new(1), true);
        let r1 = builder.or(v1, v2);
        let r3 = builder.condition(r1, VarLabel::new(1), false);
        assert!(builder.eq(r3, v1));
    }

    #[test]
    fn test_condition_compl() {
        let builder = RobddBuilder::<AllIteTable<BddPtr>>::new_with_linear_order(3);
        let v1 = builder.var(VarLabel::new(0), false);
        let v2 = builder.var(VarLabel::new(1), false);
        let r1 = builder.and(v1, v2);
        let r3 = builder.condition(r1, VarLabel::new(1), false);
        assert!(
            builder.eq(r3, v1),
            "Not eq:\nOne: {}\nTwo: {}",
            r3.to_string_debug(),
            v1.to_string_debug()
        );
    }

    #[test]
    fn test_exist() {
        let builder = RobddBuilder::<AllIteTable<BddPtr>>::new_with_linear_order(3);
        // 1 /\ 2 /\ 3
        let v1 = builder.var(VarLabel::new(0), true);
        let v2 = builder.var(VarLabel::new(1), true);
        let v3 = builder.var(VarLabel::new(2), true);
        let a1 = builder.and(v1, v2);
        let r1 = builder.and(a1, v3);
        let r_expected = builder.and(v1, v3);
        let res = builder.exists(r1, VarLabel::new(1));
        assert!(
            builder.eq(r_expected, res),
            "Got:\nOne: {}\nExpected: {}",
            res.to_string_debug(),
            r_expected.to_string_debug()
        );
    }

    #[test]
    fn test_exist_compl() {
        let builder = RobddBuilder::<AllIteTable<BddPtr>>::new_with_linear_order(3);
        // 1 /\ 2 /\ 3
        let v1 = builder.var(VarLabel::new(0), false);
        let v2 = builder.var(VarLabel::new(1), false);
        let v3 = builder.var(VarLabel::new(2), false);
        let a1 = builder.and(v1, v2);
        let r1 = builder.and(a1, v3);
        let r_expected = builder.and(v1, v3);
        let res = builder.exists(r1, VarLabel::new(1));
        // let res = r1;
        assert!(
            builder.eq(r_expected, res),
            "Got:\n: {}\nExpected: {}",
            res.to_string_debug(),
            r_expected.to_string_debug()
        );
    }

    #[test]
    fn test_compose() {
        let builder = RobddBuilder::<AllIteTable<BddPtr>>::new_with_linear_order(3);
        let v0 = builder.var(VarLabel::new(0), true);
        let v1 = builder.var(VarLabel::new(1), true);
        let v2 = builder.var(VarLabel::new(2), true);
        let v0_and_v1 = builder.and(v0, v1);
        let v0_and_v2 = builder.and(v0, v2);
        let res = builder.compose(v0_and_v1, VarLabel::new(1), v2);
        assert!(
            builder.eq(res, v0_and_v2),
            "\nGot: {}\nExpected: {}",
            res.to_string_debug(),
            v0_and_v2.to_string_debug()
        );
    }

    #[test]
    fn test_compose_2() {
        let builder = RobddBuilder::<AllIteTable<BddPtr>>::new_with_linear_order(4);
        let v0 = builder.var(VarLabel::new(0), true);
        let v1 = builder.var(VarLabel::new(1), true);
        let v2 = builder.var(VarLabel::new(2), true);
        let v3 = builder.var(VarLabel::new(3), true);
        let v0_and_v1 = builder.and(v0, v1);
        let v2_and_v3 = builder.and(v2, v3);
        let v0v2v3 = builder.and(v0, v2_and_v3);
        let res = builder.compose(v0_and_v1, VarLabel::new(1), v2_and_v3);
        assert!(
            builder.eq(res, v0v2v3),
            "\nGot: {}\nExpected: {}",
            res.to_string_debug(),
            v0v2v3.to_string_debug()
        );
    }

    #[test]
    fn test_compose_3() {
        let builder = RobddBuilder::<AllIteTable<BddPtr>>::new_with_linear_order(4);
        let v0 = builder.var(VarLabel::new(0), true);
        let v1 = builder.var(VarLabel::new(1), true);
        let v2 = builder.var(VarLabel::new(2), true);
        let f = builder.ite(v0, BddPtr::false_ptr(), v1);
        let res = builder.compose(f, VarLabel::new(1), v2);
        let expected = builder.ite(v0, BddPtr::false_ptr(), v2);
        assert!(
            builder.eq(res, expected),
            "\nGot: {}\nExpected: {}",
            res.to_string_debug(),
            expected.to_string_debug()
        );
    }

    #[test]
    fn test_compose_4() {
        let builder = RobddBuilder::<AllIteTable<BddPtr>>::new_with_linear_order(20);
        let v0 = builder.var(VarLabel::new(4), true);
        let v1 = builder.var(VarLabel::new(5), true);
        let v2 = builder.var(VarLabel::new(6), true);
        let f = builder.ite(v1, BddPtr::false_ptr(), v2);
        let res = builder.compose(f, VarLabel::new(6), v0);
        let expected = builder.ite(v1, BddPtr::false_ptr(), v0);
        assert!(
            builder.eq(res, expected),
            "\nGot: {}\nExpected: {}",
            res.to_string_debug(),
            expected.to_string_debug()
        );
    }

    #[test]
    fn test_new_label() {
        let builder = RobddBuilder::<AllIteTable<BddPtr>>::new_with_linear_order(0);
        let vlbl1 = builder.new_label();
        let vlbl2 = builder.new_label();
        let v1 = builder.var(vlbl1, false);
        let v2 = builder.var(vlbl2, false);
        let r1 = builder.and(v1, v2);
        let r3 = builder.condition(r1, VarLabel::new(1), false);
        assert!(
            builder.eq(r3, v1),
            "Not eq:\nOne: {}\nTwo: {}",
            r3.to_string_debug(),
            v1.to_string_debug()
        );
    }

    #[test]
    fn circuit1() {
        let builder = RobddBuilder::<AllIteTable<BddPtr>>::new_with_linear_order(3);
        let x = builder.var(VarLabel::new(0), false);
        let y = builder.var(VarLabel::new(1), true);
        let delta = builder.and(x, y);
        let yp = builder.var(VarLabel::new(2), true);
        let inner = builder.iff(yp, y);
        let conj = builder.and(inner, delta);
        let res = builder.exists(conj, VarLabel::new(1));

        let expected = builder.and(x, yp);
        assert!(
            builder.eq(res, expected),
            "Not eq:\nGot: {}\nExpected: {}",
            res.to_string_debug(),
            expected.to_string_debug()
        );
    }

    #[test]
    fn simple_cond() {
        let builder = RobddBuilder::<AllIteTable<BddPtr>>::new_with_linear_order(3);
        let x = builder.var(VarLabel::new(0), true);
        let y = builder.var(VarLabel::new(1), false);
        let z = builder.var(VarLabel::new(2), false);
        let r1 = builder.and(x, y);
        let r2 = builder.and(r1, z);
        // now r2 is x /\ !y /\ !z

        let res = builder.condition(r2, VarLabel::new(1), true); // condition on y=T
        let expected = BddPtr::false_ptr();
        assert!(
            builder.eq(res, expected),
            "\nOriginal BDD: {}\nNot eq:\nGot: {}\nExpected: {}",
            r2.to_string_debug(),
            res.to_string_debug(),
            expected.to_string_debug()
        );
    }

    #[test]
    fn wmc_test_2() {
        let builder = RobddBuilder::<AllIteTable<BddPtr>>::new_with_linear_order(4);
        let x = builder.var(VarLabel::new(0), true);
        let y = builder.var(VarLabel::new(1), true);
        let f1 = builder.var(VarLabel::new(2), true);
        let f2 = builder.var(VarLabel::new(3), true);

        let map = HashMap::from_iter([
            (VarLabel::new(0), (RealSemiring(1.0), RealSemiring(1.0))),
            (VarLabel::new(1), (RealSemiring(1.0), RealSemiring(1.0))),
            (VarLabel::new(2), (RealSemiring(0.8), RealSemiring(0.2))),
            (VarLabel::new(3), (RealSemiring(0.7), RealSemiring(0.3))),
        ]);

        let wmc = WmcParams::new(map);
        let iff1 = builder.iff(x, f1);
        let iff2 = builder.iff(y, f2);
        let obs = builder.or(x, y);
        let and1 = builder.and(iff1, iff2);
        let f = builder.and(and1, obs);
        assert_eq!(f.unsmoothed_wmc(&wmc).0, 0.2 * 0.3 + 0.2 * 0.7 + 0.8 * 0.3);
    }

    #[test]
    fn test_ite_1() {
        let builder = RobddBuilder::<AllIteTable<BddPtr>>::new_with_linear_order(16);
        let c1 = Cnf::from_string("(1 || 2) && (0 || -2)");
        let c2 = Cnf::from_string("(0 || 1) && (-4 || -7)");
        let cnf1 = builder.compile_cnf(&c1);
        let cnf2 = builder.compile_cnf(&c2);
        let iff1 = builder.iff(cnf1, cnf2);

        let clause1 = builder.and(cnf1, cnf2);
        let clause2 = builder.and(cnf1.neg(), cnf2.neg());
        let and = builder.or(clause1, clause2);

        if and != iff1 {
            println!("cnf1: {}", c1);
            println!("cnf2: {}", c2);
            println!(
                "not equal:\nBdd1: {}\nBdd2: {}",
                and.to_string_debug(),
                iff1.to_string_debug()
            );
        }
        assert_eq!(and, iff1);
    }

    #[test]
    fn smoothed_model_count_with_finite_field_simple() {
        static CNF: &str = "
        p cnf 3 1
        1 2 3 0
        ";
        let cnf = Cnf::from_dimacs(CNF);

        let builder = RobddBuilder::<AllIteTable<BddPtr>>::new_with_linear_order(cnf.num_vars());

        let bdd = builder.compile_cnf(&cnf);

        let smoothed = builder.smooth(bdd, cnf.num_vars());

        let weights = WmcParams::<FiniteField<1000001>>::new(HashMap::from_iter([
            (VarLabel::new(0), (FiniteField::new(1), FiniteField::new(1))),
            (VarLabel::new(1), (FiniteField::new(1), FiniteField::new(1))),
            (VarLabel::new(2), (FiniteField::new(1), FiniteField::new(1))),
        ]));

        let unsmoothed_model_count = bdd.unsmoothed_wmc(&weights);

        let smoothed_model_count = smoothed.unsmoothed_wmc(&weights);

        assert_eq!(unsmoothed_model_count.value(), 3);
        assert_eq!(smoothed_model_count.value(), 7);
    }

    #[test]
    fn smoothed_weighted_model_count_with_finite_field_simple() {
        // see: https://pysdd.readthedocs.io/en/latest/examples/model_counting.html#perform-weighted-model-counting-on-cnf-file-from-cli
        static CNF: &str = "
        p cnf 2 2
        c weights 0.4 0.6 0.3 0.7
        -1 2 0
        1 -2 0
        ";
        let cnf = Cnf::from_dimacs(CNF);

        let builder = RobddBuilder::<AllIteTable<BddPtr>>::new_with_linear_order(cnf.num_vars());

        let bdd = builder.compile_cnf(&cnf);

        let smoothed = builder.smooth(bdd, cnf.num_vars());

        let weighted_model_count =
            smoothed.unsmoothed_wmc(&WmcParams::<RealSemiring>::new(HashMap::from_iter([
                (VarLabel::new(0), (RealSemiring(0.4), RealSemiring(0.6))),
                (VarLabel::new(1), (RealSemiring(0.3), RealSemiring(0.7))),
            ])));

        assert_eq!(weighted_model_count.0, 0.54);
    }

    #[test]
    fn wmc_test_with_finite_field_complex() {
        static CNF: &str = "
        p cnf 6 3
        c weights 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60
        1 2 3 4 0
        -2 -3 4 5 0
        -4 -5 6 6 0
        ";
        let cnf = Cnf::from_dimacs(CNF);

        let builder = RobddBuilder::<AllIteTable<BddPtr>>::new_with_linear_order(cnf.num_vars());

        let bdd = builder.compile_cnf(&cnf);

        let smoothed = builder.smooth(bdd, cnf.num_vars());

        let model_count = smoothed.unsmoothed_wmc(&WmcParams::<FiniteField<1000001>>::new(
            HashMap::from_iter([
                (VarLabel::new(0), (FiniteField::new(1), FiniteField::new(1))),
                (VarLabel::new(1), (FiniteField::new(1), FiniteField::new(1))),
                (VarLabel::new(2), (FiniteField::new(1), FiniteField::new(1))),
                (VarLabel::new(3), (FiniteField::new(1), FiniteField::new(1))),
                (VarLabel::new(4), (FiniteField::new(1), FiniteField::new(1))),
                (VarLabel::new(5), (FiniteField::new(1), FiniteField::new(1))),
            ]),
        ));

        // TODO: this WMC test is broken. not sure why :(
        // let weighted_model_count = smoothed.unsmoothed_wmc(
        //     builder.order(),
        //     &WmcParams::new(HashMap::from_iter([
        //         // (VarLabel::new(0), (RealSemiring(0.10), RealSemiring(0.05))),
        //         // (VarLabel::new(1), (RealSemiring(0.20), RealSemiring(0.15))),
        //         // (VarLabel::new(2), (RealSemiring(0.30), RealSemiring(0.25))),
        //         // (VarLabel::new(3), (RealSemiring(0.40), RealSemiring(0.35))),
        //         // (VarLabel::new(4), (RealSemiring(0.50), RealSemiring(0.45))),
        //         // (VarLabel::new(5), (RealSemiring(0.60), RealSemiring(0.55))),
        //         (VarLabel::new(0), (RealSemiring(0.05), RealSemiring(0.10))),
        //         (VarLabel::new(1), (RealSemiring(0.15), RealSemiring(0.20))),
        //         (VarLabel::new(2), (RealSemiring(0.25), RealSemiring(0.30))),
        //         (VarLabel::new(3), (RealSemiring(0.35), RealSemiring(0.40))),
        //         (VarLabel::new(4), (RealSemiring(0.45), RealSemiring(0.50))),
        //         (VarLabel::new(5), (RealSemiring(0.55), RealSemiring(0.60))),
        //     ])),
        // );

        // verified with pysdd
        //
        // given tiny2-with-weights.cnf
        //
        // p cnf 6 3
        // c weights 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60
        // 1 2 3 4 0
        // -2 -3 4 5 0
        // -4 -5 6 6 0
        //
        // $ pysdd -c tiny2-with-weights.cnf
        // reading cnf...
        // Read CNF: vars=6 clauses=3
        // creating initial vtree balanced
        // creating manager...
        // compiling...

        // compilation time         : 0.001 sec
        //  sdd size                : 10
        //  sdd node count          : 5
        //  sdd model count         : 48    0.000 sec
        //  sdd weighted model count: 0.017015015625000005    0.000 sec
        // done

        assert_eq!(model_count.value(), 48);
        // assert_eq!(weighted_model_count.0, 0.017015015625000005);
    }
}
