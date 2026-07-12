//! Minimal DIMACS CNF parser (avoids external `dimacs` crate dependency).
//!
//! Supported:
//! - comment lines starting with `c`
//! - problem line `p cnf <num_vars> <num_clauses>` (optional; values are treated as hints)
//! - literals as signed integers; `0` terminates a clause; whitespace/newlines are interchangeable
//!
//! Notes:
//! - Variables are 1-indexed in DIMACS; we preserve that in the parsed output and let callers
//!   convert to 0-indexed labels if desired.

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DimacsCnf {
    pub(crate) num_vars_hint: Option<usize>,
    pub(crate) num_clauses_hint: Option<usize>,
    /// Each clause is a list of signed literals (e.g. `-3`, `5`). `0` is not included.
    pub(crate) clauses: Vec<Vec<i64>>,
}

pub(crate) fn parse_dimacs_cnf(input: &str) -> Result<DimacsCnf, String> {
    let mut num_vars_hint: Option<usize> = None;
    let mut num_clauses_hint: Option<usize> = None;

    // Collect all literal tokens after stripping comments / reading header.
    let mut toks: Vec<&str> = Vec::new();
    for line in input.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('c') {
            continue;
        }
        if line.starts_with('p') {
            // Expect: p cnf <nvars> <nclauses>
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 && parts[0] == "p" && parts[1] == "cnf" {
                num_vars_hint = parts[2]
                    .parse::<usize>()
                    .ok()
                    .or(num_vars_hint);
                num_clauses_hint = parts[3]
                    .parse::<usize>()
                    .ok()
                    .or(num_clauses_hint);
                continue;
            } else if parts.len() >= 2 && parts[0] == "p" {
                return Err(format!("unsupported DIMACS problem line: `{}`", line));
            }
        }
        toks.extend(line.split_whitespace());
    }

    let mut clauses: Vec<Vec<i64>> = Vec::new();
    let mut cur: Vec<i64> = Vec::new();

    for t in toks {
        let lit = t
            .parse::<i64>()
            .map_err(|_| format!("invalid DIMACS literal token `{}`", t))?;
        if lit == 0 {
            // End clause; allow empty clauses to represent contradiction.
            clauses.push(std::mem::take(&mut cur));
        } else {
            cur.push(lit);
        }
    }
    if !cur.is_empty() {
        return Err("DIMACS CNF ended without a terminating `0` for the final clause".to_string());
    }

    Ok(DimacsCnf {
        num_vars_hint,
        num_clauses_hint,
        clauses,
    })
}

