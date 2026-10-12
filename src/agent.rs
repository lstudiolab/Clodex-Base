//! Guarded tools for a coding agent. Operations are restricted to a workspace;
//! arbitrary shell execution is deliberately not exposed.

use anyhow::{bail, Context, Result};
use std::{
    fs,
    io::{self, Write},
    path::{Component, Path, PathBuf},
    process::Command,
};

fn canonical_workspace(workspace: &Path) -> Result<PathBuf> {
    fs::canonicalize(workspace).with_context(|| format!("workspace does not exist: {}", workspace.display()))
}

fn safe_path(workspace: &Path, relative: &Path) -> Result<PathBuf> {
    if relative.is_absolute() || relative.components().any(|c| matches!(c, Component::ParentDir | Component::Prefix(_))) {
        bail!("agent path must be relative and cannot traverse outside the workspace");
    }
    let root = canonical_workspace(workspace)?;
    let candidate = root.join(relative);
    if candidate.exists() {
        let canonical = fs::canonicalize(&candidate)?;
        if !canonical.starts_with(&root) { bail!("symlink/path escapes the workspace"); }
        Ok(canonical)
    } else {
        let parent = candidate.parent().context("path has no parent directory")?;
        let canonical_parent = fs::canonicalize(parent).with_context(|| format!("parent directory must already exist: {}", parent.display()))?;
        if !canonical_parent.starts_with(&root) { bail!("path escapes the workspace"); }
        Ok(canonical_parent.join(candidate.file_name().context("missing file name")?))
    }
}

pub fn read_file(workspace: &Path, relative: &Path) -> Result<String> {
    let path = safe_path(workspace, relative)?;
    if !path.is_file() { bail!("not a regular file: {}", relative.display()); }
    fs::read_to_string(&path).with_context(|| format!("cannot read {}", relative.display()))
}

pub fn list_files(workspace: &Path) -> Result<Vec<PathBuf>> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let s = name.to_string_lossy();
            if matches!(s.as_ref(), ".git" | ".ssh" | "target" | "build" | "node_modules" | "vendor") { continue; }
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path)?;
            if metadata.file_type().is_symlink() { continue; }
            if metadata.is_dir() { walk(root, &path, out)?; }
            else if metadata.is_file() { out.push(path.strip_prefix(root)?.to_path_buf()); }
        }
        Ok(())
    }
    let root = canonical_workspace(workspace)?;
    let mut out = Vec::new();
    walk(&root, &root, &mut out)?;
    out.sort();
    Ok(out)
}

pub fn write_file_interactive(workspace: &Path, relative: &Path, content: &str) -> Result<()> {
    let path = safe_path(workspace, relative)?;
    if !path.parent().map(Path::exists).unwrap_or(false) { bail!("target parent does not exist; create directories explicitly first"); }
    let previous = fs::read_to_string(&path).unwrap_or_default();
    println!("--- proposed change: {} ---", relative.display());
    println!("old: {} bytes | new: {} bytes", previous.len(), content.len());
    let old_lines: Vec<_> = previous.lines().collect();
    let new_lines: Vec<_> = content.lines().collect();
    for line in old_lines.iter().take(12) { println!("- {line}"); }
    for line in new_lines.iter().take(12) { println!("+ {line}"); }
    if old_lines.len() > 12 || new_lines.len() > 12 { println!("(preview truncated; full content is in the supplied content file)"); }
    print!("Apply this write? Type 'yes' to confirm: ");
    io::stdout().flush()?;
    let mut answer = String::new();
    io::stdin().read_line(&mut answer)?;
    if answer.trim() != "yes" { bail!("write cancelled; no files changed"); }
    fs::write(&path, content).with_context(|| format!("cannot write {}", relative.display()))?;
    println!("Wrote {}", relative.display());
    Ok(())
}

pub fn run_test_command(workspace: &Path, command_line: &str) -> Result<()> {
    let root = canonical_workspace(workspace)?;
    let (program, args): (&str, &[&str]) = match command_line {
        "cargo test" => ("cargo", &["test"]),
        "cargo test --workspace" => ("cargo", &["test", "--workspace"]),
        "cargo check" => ("cargo", &["check"]),
        "cmake --build build" => ("cmake", &["--build", "build"]),
        "ctest --test-dir build" => ("ctest", &["--test-dir", "build"]),
        _ => bail!("command is not allowlisted. Allowed: cargo test, cargo test --workspace, cargo check, cmake --build build, ctest --test-dir build"),
    };
    let status = Command::new(program).args(args).current_dir(&root).status()
        .with_context(|| format!("failed to launch {program}"))?;
    if !status.success() { bail!("test command failed with status {status}"); }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn path_traversal_is_rejected() {
        let root = std::env::temp_dir().join(format!("clodex-agent-test-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        assert!(safe_path(&root, Path::new("../secrets")).is_err());
        let _ = fs::remove_dir_all(root);
    }
}
