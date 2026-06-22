//! `xtask` — tri-boost's dev-only task runner (plan F0/F4).
//!
//! This crate ships **no** library code (it is `publish = false` and is not a
//! dependency of `tri-boost-core`/`tri-boost-py`), so it is in the unwrap-allowed
//! set `{tests, benches, xtask}` and does NOT inherit the workspace `[lints]`
//! panic-gate. It is pure `std`: no third-party dependencies.
//!
//! It hosts the source-scanning *grep-gates* that CI runs over the shipped crates
//! (`crates/*/src`, excluding `tests/`/`benches/` and this crate). Doing them here
//! — rather than as shell `grep` lines in `ci.yml` — keeps them block-aware,
//! cross-platform, and unit-testable:
//!
//! * `check-no-box-dyn` — no `Box<dyn Error>` in any shipped signature (§13.8 / §02.4).
//! * `check-justified` — every `unwrap`/`expect`/`panic`/`unreachable` and every form-(b)
//!   `#[allow(clippy::indexing_slicing|arithmetic_side_effects)]` in shipped, non-`#[cfg(test)]`
//!   code carries a `// JUSTIFIED:` proof (§13.8).
//! * `check-no-usize-serialized` — no `usize`/`isize` field on a serialized type (§13.4 wire-width).
//! * `check-no-hashmap-serialized` — no `HashMap`/`HashSet` field on a serialized type (order).
//! * `check-all` — run every gate; non-zero exit if any fails.
//! * `accuracy` — placeholder for the §13 accuracy harness (lands later).
//!
//! Each gate prints `file:line` for every violation and returns a non-zero
//! `ExitCode`, so CI fails closed.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// A grep-gate: scans the shipped sources and returns any violations found.
type GateFn = fn(&[SourceFile]) -> Vec<Violation>;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("--help");
    match cmd {
        "--help" | "-h" | "help" => {
            print_usage();
            ExitCode::SUCCESS
        }
        "accuracy" => {
            println!(
                "xtask accuracy: placeholder. The §13 accuracy harness (XGBoost/LightGBM/\
                 CatBoost parity on fixed datasets) lands with the learner; nothing to run in Phase 0."
            );
            ExitCode::SUCCESS
        }
        "check-no-box-dyn" => run_gate(check_no_box_dyn),
        "check-justified" => run_gate(check_justified),
        "check-no-usize-serialized" => run_gate(check_no_usize_serialized),
        "check-no-hashmap-serialized" => run_gate(check_no_hashmap_serialized),
        "check-all" => {
            let gates: [(&str, GateFn); 4] = [
                ("check-no-box-dyn", check_no_box_dyn),
                ("check-justified", check_justified),
                ("check-no-usize-serialized", check_no_usize_serialized),
                ("check-no-hashmap-serialized", check_no_hashmap_serialized),
            ];
            let files = match load_shipped_sources() {
                Ok(files) => files,
                Err(err) => {
                    eprintln!("xtask: could not enumerate shipped sources: {err}");
                    return ExitCode::FAILURE;
                }
            };
            let mut failed = false;
            for (name, gate) in gates {
                let violations = gate(&files);
                if violations.is_empty() {
                    println!("[ok]   {name}");
                } else {
                    failed = true;
                    println!("[FAIL] {name}: {} violation(s)", violations.len());
                    for v in &violations {
                        println!("    {v}");
                    }
                }
            }
            if failed {
                ExitCode::FAILURE
            } else {
                ExitCode::SUCCESS
            }
        }
        other => {
            eprintln!("xtask: unknown command `{other}`\n");
            print_usage();
            ExitCode::FAILURE
        }
    }
}

fn print_usage() {
    println!(
        "tri-boost xtask — dev-only task runner (ships no library code)\n\n\
         USAGE: cargo run -p xtask -- <command>\n\n\
         COMMANDS:\n\
         \x20 check-all                     run every grep-gate (CI entrypoint)\n\
         \x20 check-no-box-dyn              forbid `Box<dyn Error>` in shipped signatures\n\
         \x20 check-justified              require `// JUSTIFIED:` on unwrap/expect/panic/allow\n\
         \x20 check-no-usize-serialized     forbid usize/isize on serialized types\n\
         \x20 check-no-hashmap-serialized   forbid HashMap/HashSet on serialized types\n\
         \x20 accuracy                      (placeholder) accuracy harness, lands with the learner\n\
         \x20 --help                        show this message"
    );
}

/// Run one gate over the shipped sources and translate its findings into an exit code.
fn run_gate(gate: GateFn) -> ExitCode {
    let files = match load_shipped_sources() {
        Ok(files) => files,
        Err(err) => {
            eprintln!("xtask: could not enumerate shipped sources: {err}");
            return ExitCode::FAILURE;
        }
    };
    let violations = gate(&files);
    if violations.is_empty() {
        ExitCode::SUCCESS
    } else {
        for v in &violations {
            println!("{v}");
        }
        eprintln!("xtask: {} violation(s)", violations.len());
        ExitCode::FAILURE
    }
}

/// A loaded shipped source file: its display path plus its lines.
struct SourceFile {
    path: PathBuf,
    lines: Vec<String>,
}

/// One gate violation, rendered as `path:line: message`.
struct Violation {
    path: PathBuf,
    line: usize,
    message: String,
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}: {}", self.path.display(), self.line, self.message)
    }
}

/// Locate the workspace root from `CARGO_MANIFEST_DIR` (xtask's own dir → parent),
/// falling back to the current directory.
fn workspace_root() -> PathBuf {
    if let Ok(dir) = env::var("CARGO_MANIFEST_DIR") {
        if let Some(parent) = Path::new(&dir).parent() {
            return parent.to_path_buf();
        }
    }
    env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

/// Collect every shipped Rust source file (`crates/*/src/**/*.rs`), excluding any
/// `tests/`/`benches/` directory and this `xtask` crate — i.e. exactly the files
/// the no-panic / wire-width gates govern (unwrap-allowed set = `{tests, benches, xtask}`).
fn load_shipped_sources() -> std::io::Result<Vec<SourceFile>> {
    let root = workspace_root();
    let crates = root.join("crates");
    let mut rs_files = Vec::new();
    if crates.is_dir() {
        collect_rs(&crates, &mut rs_files)?;
    }
    let mut out = Vec::new();
    for path in rs_files {
        let text = fs::read_to_string(&path)?;
        let lines = text.lines().map(str::to_owned).collect();
        let display = path.strip_prefix(&root).unwrap_or(&path).to_path_buf();
        out.push(SourceFile {
            path: display,
            lines,
        });
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(out)
}

/// Recursively gather `.rs` files under `dir`, skipping `tests`/`benches` subtrees.
fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            let name = entry.file_name();
            if name == "tests" || name == "benches" || name == "target" {
                continue;
            }
            collect_rs(&path, out)?;
        } else if file_type.is_file() && path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Gate: no `Box<dyn Error>` in shipped code (§02.4 — the error model is `PbError`).
// ---------------------------------------------------------------------------

fn check_no_box_dyn(files: &[SourceFile]) -> Vec<Violation> {
    let mut out = Vec::new();
    for file in files {
        for (i, line) in file.lines.iter().enumerate() {
            let code = strip_line_comment(line);
            if code.contains("Box<dyn") && code.contains("Error") {
                out.push(Violation {
                    path: file.path.clone(),
                    line: i + 1,
                    message: "`Box<dyn Error>` is forbidden in shipped code; return `PbError`"
                        .into(),
                });
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Gate: `// JUSTIFIED:` proof required on panic-adjacent sites and form-(b) allows.
// Skips `#[cfg(test)]` modules (the unwrap-allowed set covers test code).
// ---------------------------------------------------------------------------

fn check_justified(files: &[SourceFile]) -> Vec<Violation> {
    let triggers: [&str; 6] = [
        ".unwrap(",
        ".expect(",
        "panic!",
        "unreachable!",
        "#[allow(clippy::indexing_slicing)]",
        "#[allow(clippy::arithmetic_side_effects)]",
    ];
    let mut out = Vec::new();
    for file in files {
        let in_test = test_module_mask(&file.lines);
        for (i, line) in file.lines.iter().enumerate() {
            if in_test[i] {
                continue;
            }
            let code = strip_line_comment(line);
            let trimmed = code.trim_start();
            // Skip doc/comment lines entirely (only real code triggers the gate).
            if trimmed.starts_with("//") {
                continue;
            }
            if triggers.iter().any(|t| code.contains(t)) && !has_justification(&file.lines, i) {
                out.push(Violation {
                    path: file.path.clone(),
                    line: i + 1,
                    message: "panic-adjacent site or proven-unchecked `#[allow]` lacks a `// JUSTIFIED:` proof"
                        .into(),
                });
            }
        }
    }
    out
}

/// A site is justified if a `// JUSTIFIED:` comment is on the same line or on the
/// nearest preceding non-blank line.
fn has_justification(lines: &[String], idx: usize) -> bool {
    if lines.get(idx).is_some_and(|l| l.contains("// JUSTIFIED:")) {
        return true;
    }
    let mut j = idx;
    while j > 0 {
        j -= 1;
        let prev = match lines.get(j) {
            Some(p) => p.trim(),
            None => return false,
        };
        if prev.is_empty() {
            continue;
        }
        return prev.contains("// JUSTIFIED:");
    }
    false
}

/// Mark every line that falls inside a `#[cfg(test)]` module body (brace-tracked).
/// Such lines are in the unwrap-allowed set and are exempt from `check_justified`.
fn test_module_mask(lines: &[String]) -> Vec<bool> {
    let mut mask = vec![false; lines.len()];
    let mut i = 0;
    while i < lines.len() {
        let line = match lines.get(i) {
            Some(l) => l,
            None => break,
        };
        if line.contains("#[cfg(test)]") {
            // Find the opening brace of the following `mod`, then skip to its match.
            let mut j = i;
            let mut depth = 0usize;
            let mut started = false;
            while j < lines.len() {
                if let Some(l) = lines.get(j) {
                    let opens = l.matches('{').count();
                    let closes = l.matches('}').count();
                    if opens > 0 {
                        started = true;
                    }
                    depth = depth.saturating_add(opens).saturating_sub(closes);
                    if let Some(m) = mask.get_mut(j) {
                        *m = true;
                    }
                }
                if started && depth == 0 {
                    break;
                }
                j += 1;
            }
            i = j + 1;
        } else {
            i += 1;
        }
    }
    mask
}

// ---------------------------------------------------------------------------
// Gates: forbidden field types on serialized (de)serialize-deriving items.
// ---------------------------------------------------------------------------

fn check_no_usize_serialized(files: &[SourceFile]) -> Vec<Violation> {
    serialized_field_gate(
        files,
        &["usize", "isize"],
        "`usize`/`isize` on a serialized type breaks cross-platform wire width (§13.4); use a fixed-width int",
    )
}

fn check_no_hashmap_serialized(files: &[SourceFile]) -> Vec<Violation> {
    serialized_field_gate(
        files,
        &["HashMap", "HashSet"],
        "`HashMap`/`HashSet` on a serialized type has nondeterministic order; use `BTreeMap`/`Vec`",
    )
}

/// Scan each `Serialize`/`Deserialize`-deriving struct/enum body and flag any field
/// line that mentions one of `forbidden` as a whole token.
fn serialized_field_gate(
    files: &[SourceFile],
    forbidden: &[&str],
    message: &str,
) -> Vec<Violation> {
    let mut out = Vec::new();
    for file in files {
        let mut serialized_pending = false;
        let mut i = 0;
        while i < file.lines.len() {
            let line = match file.lines.get(i) {
                Some(l) => l,
                None => break,
            };
            let code = strip_line_comment(line);
            let trimmed = code.trim_start();

            if trimmed.starts_with("#[")
                && (code.contains("Serialize") || code.contains("Deserialize"))
            {
                serialized_pending = true;
                i += 1;
                continue;
            }

            let is_item = trimmed.starts_with("struct ")
                || trimmed.starts_with("pub struct ")
                || trimmed.starts_with("pub(crate) struct ")
                || trimmed.starts_with("enum ")
                || trimmed.starts_with("pub enum ")
                || trimmed.starts_with("pub(crate) enum ");

            if is_item {
                if serialized_pending {
                    i = scan_item_body(file, i, forbidden, message, &mut out);
                    serialized_pending = false;
                    continue;
                }
                serialized_pending = false;
            } else if !trimmed.is_empty()
                && !trimmed.starts_with("//")
                && !trimmed.starts_with("#[")
            {
                // A non-attribute, non-item code line breaks the derive→item adjacency.
                serialized_pending = false;
            }
            i += 1;
        }
    }
    out
}

/// Scan a single item body starting at the item keyword line. Handles brace bodies
/// `{ .. }` and one-line tuple/unit forms ending in `;`. Returns the index just past
/// the body.
fn scan_item_body(
    file: &SourceFile,
    start: usize,
    forbidden: &[&str],
    message: &str,
    out: &mut Vec<Violation>,
) -> usize {
    // Find the first delimiter to decide brace-body vs tuple/one-liner.
    let mut i = start;
    let mut depth = 0usize;
    let mut started = false;
    while i < file.lines.len() {
        let line = match file.lines.get(i) {
            Some(l) => l,
            None => break,
        };
        let code = strip_line_comment(line);
        let opens = code.matches('{').count();
        let closes = code.matches('}').count();
        if opens > 0 {
            started = true;
        }
        if started && i > start {
            // Inside the body — check field tokens (skip the item keyword line itself).
            flag_forbidden(file, i, &code, forbidden, message, out);
        } else if i == start {
            // Tuple/unit struct on the same line, e.g. `pub struct X(pub usize);`.
            if !code.contains('{') {
                flag_forbidden(file, i, &code, forbidden, message, out);
                if code.contains(';') {
                    return i + 1;
                }
            }
        }
        depth = depth.saturating_add(opens).saturating_sub(closes);
        if started && depth == 0 {
            return i + 1;
        }
        // Tuple struct spanning to a `;` without braces.
        if !started && code.contains(';') && i > start {
            flag_forbidden(file, i, &code, forbidden, message, out);
            return i + 1;
        }
        i += 1;
    }
    i
}

/// Push a violation for each forbidden whole-token found on this line.
fn flag_forbidden(
    file: &SourceFile,
    idx: usize,
    code: &str,
    forbidden: &[&str],
    message: &str,
    out: &mut Vec<Violation>,
) {
    for tok in forbidden {
        if contains_token(code, tok) {
            out.push(Violation {
                path: file.path.clone(),
                line: idx + 1,
                message: message.into(),
            });
            break;
        }
    }
}

/// Whole-token containment: `tok` bounded by non-identifier characters (so `usize`
/// matches `n: usize` but not `my_usize_thing`).
fn contains_token(haystack: &str, tok: &str) -> bool {
    let bytes = haystack.as_bytes();
    let mut from = 0;
    while let Some(rel) = haystack.get(from..).and_then(|s| s.find(tok)) {
        let at = from + rel;
        // `map_or(true, ..)` rather than `is_none_or` — the latter is stable only
        // since Rust 1.82, above this workspace's 1.74 MSRV.
        let before_ok = at == 0 || bytes.get(at - 1).map_or(true, |b| !is_ident_byte(*b));
        let after = at + tok.len();
        let after_ok = bytes.get(after).map_or(true, |b| !is_ident_byte(*b));
        if before_ok && after_ok {
            return true;
        }
        from = at + tok.len();
    }
    false
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Strip a trailing `//` line comment (best-effort: ignores `//` inside string
/// literals, which the gates do not need to parse precisely).
fn strip_line_comment(line: &str) -> String {
    match line.find("//") {
        Some(idx) => line.get(..idx).unwrap_or("").to_string(),
        None => line.to_string(),
    }
}
