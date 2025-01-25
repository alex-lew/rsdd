use std::os::raw::c_char;
use std::{collections::HashMap, ffi::CStr};

use crate::builder::bdd::BddBuilder;
use crate::repr::DDNNFPtr;
use crate::util::semirings::{RealSemiring, RealSemiringDeriv, Semiring};
use crate::{
    builder::{bdd::RobddBuilder, cache::AllIteTable, BottomUpBuilder},
    constants::primes,
    repr::{BddPtr, Cnf, VarLabel, VarOrder, WmcParams},
    util::semirings::FiniteField,
};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[no_mangle]
pub extern "C" fn var_order_linear(num_vars: usize) -> *const VarOrder {
    Box::into_raw(Box::new(VarOrder::linear_order(num_vars)))
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn cnf_from_dimacs(dimacs_str: *const c_char) -> *const Cnf {
    let cstr = CStr::from_ptr(dimacs_str);

    Box::into_raw(Box::new(Cnf::from_dimacs(&String::from_utf8_lossy(
        cstr.to_bytes(),
    ))))
}

#[repr(C)]
pub struct WeightedSampleResult {
    sample: *mut BddPtr<'static>,
    probability: f64,
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn robdd_weighted_sample(
    builder: *mut RsddBddBuilder,
    bdd: *mut BddPtr<'static>,
    wmc_params: *mut WmcParams<RealSemiringDeriv>,
) -> WeightedSampleResult {
    if bdd.is_null() || wmc_params.is_null() {
        eprintln!("Fatal error, got NULL pointer for `bdd` or `wmc_params`");
        std::process::abort();
    }

    let builder = robdd_builder_from_ptr(builder);
    let bdd = *bdd;
    let wmc_params = &*wmc_params;

    let (sample, sample_probability) = builder.weighted_sample(bdd, wmc_params);
    WeightedSampleResult {
        sample: Box::into_raw(Box::new(sample)),
        probability: sample_probability,
    }
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn robdd_top_k_paths(
    builder: *mut RsddBddBuilder,
    bdd: *mut BddPtr<'static>,
    k: usize,
    wmc: *mut WmcParams<RealSemiringDeriv>,
) -> *mut BddPtr<'static> {
    let builder = robdd_builder_from_ptr(builder);
    let bdd = *bdd;
    let wmc = &*wmc;
    let sample = builder.top_k_paths(bdd, k, wmc);
    Box::into_raw(Box::new(sample))
}

// directly inspired by https://users.rust-lang.org/t/how-to-deal-with-lifetime-when-need-to-expose-through-ffi/39583
// and the follow-up at https://users.rust-lang.org/t/can-someone-explain-why-this-is-working/82324/6
#[repr(C)]
pub struct RsddBddBuilder {
    _priv: [u8; 0],
}

unsafe fn robdd_builder_from_ptr<'_0>(
    ptr: *mut RsddBddBuilder,
) -> &'_0 mut RobddBuilder<'static, AllIteTable<BddPtr<'static>>> {
    if ptr.is_null() {
        eprintln!("Fatal error, got NULL `Context` pointer");
        ::std::process::abort();
    }
    &mut *(ptr.cast())
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn robdd_builder_all_table(order: *mut VarOrder) -> *mut RsddBddBuilder {
    if order.is_null() {
        eprintln!("Fatal error, got NULL `order` pointer");
        std::process::abort();
    }

    let order = *Box::from_raw(order);
    Box::into_raw(Box::new(RobddBuilder::<AllIteTable<BddPtr>>::new(order, None))).cast()
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn robdd_builder_compile_cnf(
    builder: *mut RsddBddBuilder,
    cnf: *mut Cnf,
) -> *mut BddPtr<'static> {
    if cnf.is_null() {
        eprintln!("Fatal error, got NULL `cnf` pointer");
        std::process::abort();
    }

    let builder = robdd_builder_from_ptr(builder);
    let cnf = *Box::from_raw(cnf);
    let ptr = builder.compile_cnf(&cnf);
    Box::into_raw(Box::new(ptr))
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn robdd_model_count(
    builder: *mut RsddBddBuilder,
    bdd: *mut BddPtr<'static>,
) -> u64 {
    let builder = robdd_builder_from_ptr(builder);
    let num_vars = builder.num_vars();
    let smoothed = builder.smooth(*bdd, num_vars);
    let unweighted_params: WmcParams<FiniteField<{ primes::U64_LARGEST }>> =
        WmcParams::new(HashMap::from_iter(
            (0..num_vars as u64)
                .map(|v| (VarLabel::new(v), (FiniteField::one(), FiniteField::one()))),
        ));

    let mc = smoothed.unsmoothed_wmc(&unweighted_params).value();
    mc as u64
}

// implementing the disc interface

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn mk_bdd_manager_default_order(num_vars: u64) -> *mut RsddBddBuilder {
    Box::into_raw(Box::new(RobddBuilder::<AllIteTable<BddPtr>>::new(
        VarOrder::linear_order(num_vars as usize), None
    )))
    .cast()
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn start_bdd_manager_time_limit(builder: *mut RsddBddBuilder, time_limit: f64) {
    let duration = std::time::Duration::from_secs_f64(time_limit);
    let builder = robdd_builder_from_ptr(builder);
    builder.start_time_limit(duration);
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn stop_bdd_manager_time_limit(builder: *mut RsddBddBuilder) {
    let builder = robdd_builder_from_ptr(builder);
    builder.stop_time_limit();
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn bdd_manager_time_limit_exceeded(builder: *mut RsddBddBuilder) -> bool {
    let builder = robdd_builder_from_ptr(builder);
    builder.check_time_limit()
}


#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn bdd_new_label(builder: *mut RsddBddBuilder) -> u64 {
    let builder = robdd_builder_from_ptr(builder);
    builder.new_label().value()
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn bdd_var(
    builder: *mut RsddBddBuilder,
    label: u64,
    polarity: bool,
) -> *mut BddPtr<'static> {
    let builder = robdd_builder_from_ptr(builder);
    let ptr = builder.var(VarLabel::new(label), polarity);
    Box::into_raw(Box::new(ptr))
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn bdd_new_var(
    builder: *mut RsddBddBuilder,
    polarity: bool,
) -> *mut BddPtr<'static> {
    let builder = robdd_builder_from_ptr(builder);
    let (_, ptr) = builder.new_var(polarity);
    Box::into_raw(Box::new(ptr))
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn bdd_new_var_at_position(
    builder: *mut RsddBddBuilder,
    position: usize,
    polarity: bool,
) -> *mut BddPtr<'static> {
    let builder = robdd_builder_from_ptr(builder);
    let (_, ptr) = builder.new_var_at_position(position, polarity);
    Box::into_raw(Box::new(ptr))
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn bdd_ite(
    builder: *mut RsddBddBuilder,
    f: *mut BddPtr<'static>,
    g: *mut BddPtr<'static>,
    h: *mut BddPtr<'static>,
) -> *mut BddPtr<'static> {
    let builder = robdd_builder_from_ptr(builder);
    let and = builder.ite(*f, *g, *h);
    Box::into_raw(Box::new(and))
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn bdd_condition(
    builder: *mut RsddBddBuilder,
    bdd: *mut BddPtr<'static>,
    label: u64,
    value: bool,
) -> *mut BddPtr<'static> {
    let builder = robdd_builder_from_ptr(builder);
    let conditioned = builder.condition(*bdd, VarLabel::new(label), value);
    Box::into_raw(Box::new(conditioned))
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn bdd_and(
    builder: *mut RsddBddBuilder,
    left: *mut BddPtr<'static>,
    right: *mut BddPtr<'static>,
) -> *mut BddPtr<'static> {
    let builder = robdd_builder_from_ptr(builder);
    let and = builder.and(*left, *right);
    Box::into_raw(Box::new(and))
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn bdd_or(
    builder: *mut RsddBddBuilder,
    left: *mut BddPtr<'static>,
    right: *mut BddPtr<'static>,
) -> *mut BddPtr<'static> {
    let builder = robdd_builder_from_ptr(builder);
    let or = builder.or(*left, *right);
    Box::into_raw(Box::new(or))
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn bdd_negate(
    builder: *mut RsddBddBuilder,
    bdd: *mut BddPtr<'static>,
) -> *mut BddPtr<'static> {
    let builder = robdd_builder_from_ptr(builder);
    let negate = builder.negate(*bdd);
    Box::into_raw(Box::new(negate))
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn bdd_is_true(bdd: *mut BddPtr<'static>) -> bool {
    (*bdd).is_true()
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn bdd_is_false(bdd: *mut BddPtr<'static>) -> bool {
    (*bdd).is_false()
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn bdd_is_const(bdd: *mut BddPtr<'static>) -> bool {
    (*bdd).is_const()
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn bdd_true(builder: *mut RsddBddBuilder) -> *mut BddPtr<'static> {
    let builder = robdd_builder_from_ptr(builder);
    let bdd = builder.true_ptr();
    Box::into_raw(Box::new(bdd))
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn bdd_false(builder: *mut RsddBddBuilder) -> *mut BddPtr<'static> {
    let builder = robdd_builder_from_ptr(builder);
    let bdd = builder.false_ptr();
    Box::into_raw(Box::new(bdd))
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn bdd_eq(
    builder: *mut RsddBddBuilder,
    left: *mut BddPtr<'static>,
    right: *mut BddPtr<'static>,
) -> bool {
    let builder = robdd_builder_from_ptr(builder);
    builder.eq(*left, *right)
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn bdd_topvar(bdd: *mut BddPtr<'static>) -> u64 {
    match (*bdd).var_safe() {
        Some(x) => x.value(),
        None => 0, // TODO: fix this
    }
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn bdd_low(bdd: *mut BddPtr<'static>) -> *mut BddPtr<'static> {
    Box::into_raw(Box::new((*bdd).low()))
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn bdd_high(bdd: *mut BddPtr<'static>) -> *mut BddPtr<'static> {
    Box::into_raw(Box::new((*bdd).high()))
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn print_bdd(bdd: *mut BddPtr<'static>) -> *const c_char {
    let s = std::ffi::CString::new((*bdd).print_bdd()).unwrap();
    let p = s.as_ptr();
    std::mem::forget(s);
    p
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn bdd_json(bdd: *mut BddPtr<'static>) -> *const c_char {
    let s = std::ffi::CString::new((*bdd).bdd_json()).unwrap();
    let p = s.as_ptr();
    std::mem::forget(s);
    p
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn bdd_num_recursive_calls(builder: *mut RsddBddBuilder) -> usize {
    let builder = robdd_builder_from_ptr(builder);
    builder.num_recursive_calls()
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn bdd_wmc(
    bdd: *mut BddPtr<'static>,
    wmc: *mut WmcParams<RealSemiringDeriv>,
) -> f64 {
    DDNNFPtr::unsmoothed_wmc(&(*bdd), &(*wmc)).0
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn new_wmc_params_f64() -> *mut WmcParams<RealSemiringDeriv> {
    Box::into_raw(Box::new(WmcParams::new(HashMap::from([]))))
}
#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn bdd_compose(
    builder: *mut RsddBddBuilder,
    f: *mut BddPtr<'static>,
    var: u64,
    g: *mut BddPtr<'static>,
) -> *mut BddPtr<'static> {
    let builder = robdd_builder_from_ptr(builder);
    let result = builder.compose(*f, VarLabel::new(var), *g);
    Box::into_raw(Box::new(result))
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn bdd_exists(
    builder: *mut RsddBddBuilder,
    f: *mut BddPtr<'static>,
    var: u64,
) -> *mut BddPtr<'static> {
    let builder = robdd_builder_from_ptr(builder);
    let result = builder.exists(*f, VarLabel::new(var));
    Box::into_raw(Box::new(result))
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn wmc_param_f64_set_weight(
    weights: *mut WmcParams<RealSemiringDeriv>,
    var: u64,
    low: f64,
    high: f64,
) {
    (*weights).set_weight(VarLabel::new(var), RealSemiringDeriv(low, 0.0), RealSemiringDeriv(high, 0.0))
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct WeightF64(pub f64, pub f64);

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn wmc_param_f64_var_weight(
    weights: *mut WmcParams<RealSemiringDeriv>,
    var: u64,
) -> WeightF64 {
    let (l, h) = (*weights).var_weight(VarLabel::new(var));
    WeightF64(l.0, h.0)
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn weight_f64_lo(w: WeightF64) -> f64 {
    w.0
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn weight_f64_hi(w: WeightF64) -> f64 {
    w.1
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn bdd_size(bdd: *mut BddPtr<'static>) -> usize {
    (*bdd).count_nodes()
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn bdd_iff(
    builder: *mut RsddBddBuilder,
    left: *mut BddPtr<'static>,
    right: *mut BddPtr<'static>,
) -> *mut BddPtr<'static> {
    let builder = robdd_builder_from_ptr(builder);
    let iff = builder.iff(*left, *right);
    Box::into_raw(Box::new(iff))
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn bdd_has_variable(
    builder: *mut RsddBddBuilder,
    bdd: *mut BddPtr<'static>,
    var: u64,
) -> bool {
    let builder = robdd_builder_from_ptr(builder);
    builder.has_variable(*bdd, VarLabel::new(var))
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn bdd_hash(bdd: *mut BddPtr<'static>) -> u64 {
    let bdd = *bdd;

    let mut hasher = DefaultHasher::new();
    bdd.hash(&mut hasher);
    hasher.finish()
}

#[no_mangle]
pub unsafe extern "C" fn free_bdd(bdd: *mut BddPtr<'static>) {
    if !bdd.is_null() {
        drop(Box::from_raw(bdd));
    }
}

#[no_mangle]
pub unsafe extern "C" fn free_bdd_manager(manager: *mut RsddBddBuilder) {
    if !manager.is_null() {
        drop(Box::from_raw(
            manager.cast::<RobddBuilder<AllIteTable<BddPtr>>>(),
        ));
    }
}

#[no_mangle]
pub unsafe extern "C" fn free_wmc_params(params: *mut WmcParams<RealSemiringDeriv>) {
    if !params.is_null() {
        drop(Box::from_raw(params));
    }
}
