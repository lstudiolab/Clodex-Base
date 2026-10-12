//! Guarded tools for a coding agent. Operations are restricted to a workspace;
//! arbitrary shell execution is deliberately not exposed.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::{
    fs,
    io::{self, Write},
    path::{Component, Path, PathBuf},
    process::Command,
    thread,
    time::{Duration, Instant},
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
    let previous = if path.is_file() {
        fs::read_to_string(&path).with_context(|| format!("existing target is not a readable UTF-8 text file: {}", relative.display()))?
    } else {
        String::new()
    };
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
    let mut child = Command::new(program).args(args).current_dir(&root).spawn()
        .with_context(|| format!("failed to launch {program}"))?;
    let deadline = Instant::now() + Duration::from_secs(900);
    loop {
        if let Some(status) = child.try_wait().context("failed while waiting for test process")? {
            if !status.success() { bail!("test command failed with status {status}"); }
            return Ok(());
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            bail!("test command exceeded the 15-minute safety timeout");
        }
        thread::sleep(Duration::from_millis(100));
    }
}


#[derive(Debug, Deserialize)]
#[serde(tag = "tool", rename_all = "snake_case")]
enum ToolCall {
    ListFiles,
    ReadFile { path: String },
    SearchFiles { query: String },
    WriteFile { path: String, content: String },
    RunTests { command: String },
    Finish { summary: String },
}

fn parse_tool_call(raw: &str) -> Result<ToolCall> {
    let start = raw.find('{').context("model output did not contain a JSON object")?;
    let candidate = &raw[start..];
    let mut decoder = serde_json::Deserializer::from_str(candidate);
    let call = ToolCall::deserialize(&mut decoder).context("model output was not a valid supported tool-call object")?;
    decoder.end().context("model output contained trailing text after the tool-call object")?;
    Ok(call)
}

fn truncate_observation(text: String, max_bytes: usize) -> String {
    if text.len() <= max_bytes { return text; }
    let mut end = max_bytes.min(text.len());
    while !text.is_char_boundary(end) { end -= 1; }
    format!("{}\n[observation truncated]", &text[..end])
}

fn search_workspace(workspace: &Path, query: &str) -> Result<String> {
    if query.trim().is_empty() { bail!("search query cannot be empty"); }
    let needle = query.to_lowercase();
    let mut hits = Vec::new();
    for relative in list_files(workspace)? {
        let path = safe_path(workspace, &relative)?;
        let metadata = fs::metadata(&path)?;
        if metadata.len() > 1_000_000 { continue; }
        let Ok(contents) = fs::read_to_string(&path) else { continue; };
        for (index, line) in contents.lines().enumerate() {
            if line.to_lowercase().contains(&needle) {
                hits.push(format!("{}:{}: {}", relative.display(), index + 1, line.trim()));
                if hits.len() >= 20 { return Ok(hits.join("\n")); }
            }
        }
    }
    if hits.is_empty() { Ok("No matches found in readable files under 1 MB.".into()) }
    else { Ok(hits.join("\n")) }
}

/// Runs the tool-use loop around the configured local model. Model output is untrusted;
/// only valid structured actions reach these restricted tools.
pub fn run_model_agent<F>(
    workspace: &Path,
    task: &str,
    max_iterations: usize,
    response_bytes: usize,
    mut generate: F,
) -> Result<()>
where
    F: FnMut(&str, usize) -> Result<String>,
{
    if task.trim().is_empty() { bail!("task cannot be empty"); }
    if !(1..=20).contains(&max_iterations) { bail!("iterations must be between 1 and 20"); }
    if !(32..=8192).contains(&response_bytes) { bail!("response-bytes must be between 32 and 8192"); }
    let root = canonical_workspace(workspace)?;
    let mut history: Vec<String> = Vec::new();

    for turn in 1..=max_iterations {
        let recent = history.iter().rev().take(6).rev().cloned().collect::<Vec<_>>().join("\n");
        let prompt = format!(
            "You are Clodex, a cautious coding agent. Work only inside this workspace: {}.\n\
             User task: {}\n\
             Available actions: {{\"tool\":\"list_files\"}}, {{\"tool\":\"read_file\",\"path\":\"relative/path\"}}, \
             {{\"tool\":\"search_files\",\"query\":\"text\"}}, {{\"tool\":\"write_file\",\"path\":\"relative/path\",\"content\":\"complete file contents\"}}, \
             {{\"tool\":\"run_tests\",\"command\":\"cargo test\"}}, {{\"tool\":\"finish\",\"summary\":\"result\"}}.\n\
             Respond with exactly one JSON object and no markdown. Never request shell commands other than the listed test commands.\n\
             Previous observations:\n{}\nNext action JSON:",
            root.display(), task, recent
        );
        let raw = generate(&prompt, response_bytes)?;
        let call = match parse_tool_call(&raw) {
            Ok(call) => call,
            Err(error) => {
                let observation = format!("turn {turn}: invalid tool call ({error}). Return one valid JSON action object only.");
                eprintln!("{observation}");
                history.push(observation);
                continue;
            }
        };

        let observation = match call {
            ToolCall::ListFiles => {
                let files = list_files(&root)?;
                format!("Listed {} files:\n{}", files.len(), files.iter().take(200).map(|p| p.display().to_string()).collect::<Vec<_>>().join("\n"))
            }
            ToolCall::ReadFile { path } => match read_file(&root, Path::new(&path)) {
                Ok(contents) => format!("FILE {path}:\n{}", truncate_observation(contents, 12_000)),
                Err(error) => format!("Read failed: {error}"),
            },
            ToolCall::SearchFiles { query } => match search_workspace(&root, &query) {
                Ok(result) => format!("Search results for {query:?}:\n{result}"),
                Err(error) => format!("Search failed: {error}"),
            },
            ToolCall::WriteFile { path, content } => {
                if content.len() > 256_000 {
                    "Write rejected: one tool call cannot write more than 256 KB.".into()
                } else {
                    match write_file_interactive(&root, Path::new(&path), &content) {
                        Ok(()) => format!("Write applied to {path}. Run relevant tests next."),
                        Err(error) => format!("Write not applied: {error}"),
                    }
                }
            }
            ToolCall::RunTests { command } => match run_test_command(&root, &command) {
                Ok(()) => format!("Test command succeeded: {command}"),
                Err(error) => format!("Test command failed or was rejected: {error}"),
            },
            ToolCall::Finish { summary } => {
                println!("Clodex task result: {}", truncate_observation(summary, 4_000));
                return Ok(());
            }
        };
        println!("Agent turn {turn}/{max_iterations}: {}", truncate_observation(observation.clone(), 1_500));
        history.push(format!("turn {turn}: {}", truncate_observation(observation, 3_000)));
    }
    println!("Clodex stopped at the iteration limit. Review the observations above; no additional action was taken.");
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

    #[test]
    fn tool_call_schema_rejects_unknown_tools() {
        assert!(parse_tool_call(r#"{"tool":"rm","command":"rm -rf /"}"#).is_err());
        assert!(matches!(parse_tool_call(r#"{"tool":"finish","summary":"done"}"#).unwrap(), ToolCall::Finish { .. }));
    }

    #[test]
    fn model_agent_executes_a_structured_finish_action() {
        let root = std::env::temp_dir().join(format!("clodex-loop-test-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let mut invoked = false;
        run_model_agent(&root, "inspect only", 1, 128, |prompt, limit| {
            invoked = true;
            assert!(prompt.contains("inspect only"));
            assert_eq!(limit, 128);
            Ok(r#"{"tool":"finish","summary":"test complete"}"#.to_owned())
        }).unwrap();
        assert!(invoked);
        let _ = fs::remove_dir_all(root);
    }
}
