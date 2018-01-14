use apply_cache::*;
use bdd::*;

const INITIAL_CAPACITY: usize = 19; // given as a power of two

#[derive(Debug, PartialEq, Clone)]
pub struct BddCacheStats {
    pub lookup_count: usize,
    pub miss_count: usize,
    pub conflict_count: usize,
    pub avg_probe: f64,
    pub num_applications: usize,
}

impl BddCacheStats {
    pub fn new() -> BddCacheStats {
        BddCacheStats {
            miss_count: 0,
            lookup_count: 0,
            avg_probe: 0.0,
            num_applications: 0,
            conflict_count: 0,
        }
    }
}

/// The top-level data structure which caches applications
pub struct BddApplyTable {
    /// a table of Ite triples
    table: Vec<SubTable<(BddPtr, BddPtr), BddPtr>>,
}

impl BddApplyTable {
    pub fn new(num_vars: usize) -> BddApplyTable {
        let mut tbl = BddApplyTable {
            table: Vec::with_capacity(num_vars),
        };
        for _ in 0..num_vars {
            tbl.table.push(SubTable::new(INITIAL_CAPACITY));
        }
        tbl
    }

    /// Insert an operation into the apply table. Note that operations are
    /// normalized by first sorting the sub-BDDs such that BDD A occurs first
    /// in the ordering; this increases cache hit rate and decreases duplicate
    /// storage
    pub fn insert(&mut self, f: BddPtr, g: BddPtr, res: BddPtr) -> () {
        let tbl = f.var() as usize;
        self.table[tbl].insert((f, g), res);
    }

    pub fn get(&mut self, f: BddPtr, g: BddPtr) -> Option<BddPtr> {
        let tbl = f.var() as usize;
        self.table[tbl].get((f, g))
    }

    pub fn get_stats(&self) -> BddCacheStats {
        let mut st = BddCacheStats::new();
        for tbl in self.table.iter() {
            let stats = tbl.get_stats();
            st.lookup_count += stats.lookup_count;
            st.miss_count += stats.miss_count;
            st.conflict_count += stats.conflict_count;
            st.num_applications += tbl.len();
        }
        st
    }
}

#[test]
fn apply_cache_simple() {
    let mut tbl = BddApplyTable::new(11);
    for var in 0..10 {
        for i in 0..100000 {
            let f = BddPtr::new(VarLabel::new(var), TableIndex::new(i));
            let g = BddPtr::new(VarLabel::new(var + 1), TableIndex::new(i));
            let result = BddPtr::new(VarLabel::new(var), TableIndex::new(i));
            tbl.insert(f, g, result);
        }
    }
    for var in 0..10 {
        for i in 0..100000 {
            let f = BddPtr::new(VarLabel::new(var), TableIndex::new(i));
            let g = BddPtr::new(VarLabel::new(var + 1), TableIndex::new(i));
            let result = BddPtr::new(VarLabel::new(var), TableIndex::new(i));
            tbl.insert(f, g, result);
            assert_eq!(tbl.get(f, g).unwrap(), result);
        }
    }
}
