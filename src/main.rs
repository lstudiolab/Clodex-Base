mod agent;
mod tokenizer;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    ffi::{CStr, CString},
    fs,
    io::{self, Read, Write},
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

const DEFAULT_CHECKPOINT: &str = "artifacts/clodex.latest.ckpt";

unsafe extern "C" {
    fn clodex_train(
        corpus_path: *const std::ffi::c_char,
        checkpoint_path: *const std::ffi::c_char,
        steps: i32,
        sequence_length: i32,
        hidden_size: i32,
        learning_rate: f64,
        resume: i32,
        save_every: i32,
    ) -> i32;
    fn clodex_generate(
        checkpoint_path: *const std::ffi::c_char,
        prompt: *const u8,
        prompt_size: usize,
        max_new_tokens: usize,
        seed: u64,
        temperature: f64,
        output: *mut u8,
        output_capacity: usize,
        output_size: *mut usize,
    ) -> i32;
    fn clodex_last_error() -> *const std::ffi::c_char;
    fn clodex_engine_version() -> *const std::ffi::c_char;
}

#[derive(Parser)]
#[command(name = "clodex", version, about = "Clodex — trainable language model and guarded coding agent")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train or continue training from the latest checkpoint.
    Train {
        #[arg(long, default_value = "data/train.txt")]
        corpus: PathBuf,
        #[arg(long, default_value = DEFAULT_CHECKPOINT)]
        checkpoint: PathBuf,
        #[arg(long, default_value_t = 1000)]
        steps: i32,
        #[arg(long, default_value_t = 64)]
        sequence_length: i32,
        #[arg(long, default_value_t = 128)]
        hidden_size: i32,
        #[arg(long, default_value_t = 0.001)]
        learning_rate: f64,
        #[arg(long, default_value_t = 100)]
        save_every: i32,
        /// Start from random weights instead of resuming an existing checkpoint.
        #[arg(long, default_value_t = false)]
        fresh: bool,
        /// Upload the completed checkpoint to the private repository release using authenticated GitHub CLI.
        #[arg(long, default_value_t = false)]
        publish_release: bool,
    },
    /// Generate a continuation using a trained checkpoint.
    Generate {
        #[arg(long, default_value = DEFAULT_CHECKPOINT)]
        checkpoint: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 256)]
        max_new_tokens: usize,
        #[arg(long, default_value_t = 0.8)]
        temperature: f64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Build and exercise the guarded coding-agent tools.
    Agent {
        #[command(subcommand)]
        command: AgentCommands,
    },
    /// Print runtime versions and the current training/checkpoint status.
    Status {
        #[arg(long, default_value = DEFAULT_CHECKPOINT)]
        checkpoint: PathBuf,
        #[arg(long, default_value = "data/train.txt")]
        corpus: PathBuf,
    },
    /// Verify and fingerprint a text corpus without changing it.
    InspectCorpus {
        #[arg(long, default_value = "data/train.txt")]
        path: PathBuf,
    },
}

#[derive(Subcommand)]
enum AgentCommands {
    /// Read a project file after enforcing workspace boundaries.
    Read { workspace: PathBuf, path: PathBuf },
    /// Write a file only after showing a diff preview and receiving confirmation.
    Write { workspace: PathBuf, path: PathBuf, content_file: PathBuf },
    /// List files in a workspace (hidden build and VCS folders are excluded).
    List { workspace: PathBuf },
    /// Run a known project test command with explicit allowlisting.
    Test { workspace: PathBuf, #[arg(long, default_value = "cargo test")] command: String },
}

#[derive(Serialize, Deserialize)]
struct RunManifest {
    format_version: u32,
    model: String,
    tokenizer: String,
    corpus_sha256: String,
    started_unix_seconds: u64,
    requested_steps: i32,
    checkpoint_path: String,
    implementation: String,
    notes: Vec<String>,
}

fn cstring_path(path: &Path) -> Result<CString> {
    CString::new(path.to_string_lossy().as_bytes()).context("path contains an embedded NUL")
}

fn engine_error() -> String {
    unsafe {
        let ptr = clodex_last_error();
        if ptr.is_null() { "unknown C++ engine error".into() }
        else { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
    }
}

fn hash_file(path: &Path) -> Result<String> {
    let mut f = fs::File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 { break; }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}

fn current_time() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

fn try_fetch_latest_release(checkpoint: &Path) -> bool {
    if checkpoint.exists() { return true; }
    let parent = checkpoint.parent().unwrap_or_else(|| Path::new("."));
    if fs::create_dir_all(parent).is_err() { return false; }
    let result = Command::new("gh")
        .args(["release", "download", "clodex-latest", "--repo", "lstudiolab/Clodex-Base", "--pattern", "clodex.latest.ckpt", "--dir"])
        .arg(parent)
        .status();
    match result {
        Ok(status) if status.success() => {
            let downloaded = parent.join("clodex.latest.ckpt");
            if downloaded.is_file() && downloaded != checkpoint {
                if fs::rename(&downloaded, checkpoint).is_err() {
                    return fs::copy(&downloaded, checkpoint).is_ok();
                }
            }
            checkpoint.is_file()
        }
        _ => false,
    }
}

fn publish_latest_release(checkpoint: &Path, manifest: &Path) -> Result<()> {
    let staging = std::env::temp_dir().join(format!("clodex-release-{}", std::process::id()));
    fs::create_dir_all(&staging)?;
    let staged_checkpoint = staging.join("clodex.latest.ckpt");
    let staged_manifest = staging.join("clodex.latest.run.json");
    fs::copy(checkpoint, &staged_checkpoint)?;
    if manifest.is_file() { fs::copy(manifest, &staged_manifest)?; }

    let repo = "lstudiolab/Clodex-Base";
    let tag = "clodex-latest";
    let release_exists = Command::new("gh").args(["release", "view", tag, "--repo", repo]).status()
        .map(|status| status.success()).unwrap_or(false);
    let result = if release_exists {
        let mut cmd = Command::new("gh");
        cmd.args(["release", "upload", tag, "--repo", repo, "--clobber"])
            .arg(&staged_checkpoint);
        if staged_manifest.is_file() { cmd.arg(&staged_manifest); }
        cmd.status().context("failed to start GitHub CLI while uploading checkpoint")?
    } else {
        let mut cmd = Command::new("gh");
        cmd.args(["release", "create", tag, "--repo", repo, "--title", "Clodex latest checkpoint", "--notes", "Latest resumable Clodex training checkpoint. This release asset is updated only when a training run explicitly uses --publish-release."]);
        cmd.arg(&staged_checkpoint);
        if staged_manifest.is_file() { cmd.arg(&staged_manifest); }
        cmd.status().context("failed to start GitHub CLI while creating checkpoint release")?
    };
    let _ = fs::remove_dir_all(&staging);
    if !result.success() {
        bail!("GitHub Release upload failed. The local checkpoint remains saved. Check gh auth status and repository permissions.");
    }
    println!("Published checkpoint to https://github.com/{repo}/releases/tag/{tag}");
    Ok(())
}

fn run_train(corpus: PathBuf, checkpoint: PathBuf, steps: i32, sequence_length: i32, hidden_size: i32, learning_rate: f64, save_every: i32, fresh: bool, publish_release: bool) -> Result<()> {
    if steps <= 0 || sequence_length < 2 || hidden_size < 8 || save_every <= 0 {
        bail!("steps/save_every must be positive, sequence length at least 2, and hidden size at least 8");
    }
    if !(learning_rate.is_finite() && learning_rate > 0.0 && learning_rate <= 1.0) {
        bail!("learning rate must be finite and in (0, 1]");
    }
    if !fresh && !checkpoint.exists() {
        if try_fetch_latest_release(&checkpoint) {
            println!("Fetched latest checkpoint from the Clodex GitHub Release.");
        } else {
            println!("No local checkpoint or accessible clodex-latest release checkpoint found; starting fresh.");
        }
    }
    let metadata = fs::metadata(&corpus).with_context(|| format!("training corpus not found: {}. Add plain-text training data first; no dataset is bundled.", corpus.display()))?;
    if metadata.len() < sequence_length as u64 + 2 {
        bail!("corpus is too small for the requested sequence length");
    }
    if let Some(parent) = checkpoint.parent() { fs::create_dir_all(parent)?; }
    let run = RunManifest {
        format_version: 1,
        model: "clodex-byte-rnn-v1".into(),
        tokenizer: "utf8-byte-ids-v1 (256 byte tokens + BOS + EOS)".into(),
        corpus_sha256: hash_file(&corpus)?,
        started_unix_seconds: current_time(),
        requested_steps: steps,
        checkpoint_path: checkpoint.to_string_lossy().into_owned(),
        implementation: "Rust control plane + C++ CPU engine".into(),
        notes: vec![
            "Checkpoint resume restores model weights, optimizer state, and step count.".into(),
            "This baseline is a compact recurrent language model, not a frontier-scale transformer.".into(),
            "Keep datasets and checkpoints outside Git history; publish checkpoints as release assets.".into(),
        ],
    };
    let manifest_path = checkpoint.with_extension("run.json");
    fs::write(&manifest_path, serde_json::to_vec_pretty(&run)?)?;
    let corpus_c = cstring_path(&corpus)?;
    let checkpoint_c = cstring_path(&checkpoint)?;
    println!("Clodex engine: {}", unsafe { CStr::from_ptr(clodex_engine_version()).to_string_lossy() });
    println!("Corpus: {} ({} bytes, SHA-256 {})", corpus.display(), metadata.len(), run.corpus_sha256);
    println!("Checkpoint: {} | mode: {}", checkpoint.display(), if fresh { "fresh" } else { "resume if compatible" });
    let code = unsafe {
        clodex_train(corpus_c.as_ptr(), checkpoint_c.as_ptr(), steps, sequence_length, hidden_size, learning_rate, if fresh { 0 } else { 1 }, save_every)
    };
    if code != 0 { bail!("training failed: {}", engine_error()); }
    println!("Training session finished. Latest checkpoint saved to {}.", checkpoint.display());
    if publish_release { publish_latest_release(&checkpoint, &manifest_path)?; }
    Ok(())
}

fn run_generate(checkpoint: PathBuf, prompt: String, max_new_tokens: usize, temperature: f64, seed: u64) -> Result<()> {
    if max_new_tokens == 0 || max_new_tokens > 16_384 { bail!("max-new-tokens must be between 1 and 16384"); }
    if !(temperature.is_finite() && (0.05..=3.0).contains(&temperature)) { bail!("temperature must be in [0.05, 3.0]"); }
    if !checkpoint.is_file() { bail!("checkpoint not found: {}. Train the model first.", checkpoint.display()); }
    let checkpoint_c = cstring_path(&checkpoint)?;
    let mut out = vec![0u8; max_new_tokens];
    let mut written = 0usize;
    let code = unsafe { clodex_generate(checkpoint_c.as_ptr(), prompt.as_bytes().as_ptr(), prompt.len(), max_new_tokens, seed, temperature, out.as_mut_ptr(), out.len(), &mut written) };
    if code != 0 { bail!("generation failed: {}", engine_error()); }
    io::stdout().write_all(prompt.as_bytes())?;
    io::stdout().write_all(&out[..written])?;
    io::stdout().flush()?;
    println!();
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Train { corpus, checkpoint, steps, sequence_length, hidden_size, learning_rate, save_every, fresh, publish_release } =>
            run_train(corpus, checkpoint, steps, sequence_length, hidden_size, learning_rate, save_every, fresh, publish_release),
        Commands::Generate { checkpoint, prompt, max_new_tokens, temperature, seed } =>
            run_generate(checkpoint, prompt, max_new_tokens, temperature, seed),
        Commands::Agent { command } => match command {
            AgentCommands::Read { workspace, path } => { print!("{}", agent::read_file(&workspace, &path)?); Ok(()) }
            AgentCommands::Write { workspace, path, content_file } => {
                let content = fs::read_to_string(&content_file).with_context(|| format!("cannot read content file {}", content_file.display()))?;
                agent::write_file_interactive(&workspace, &path, &content)
            }
            AgentCommands::List { workspace } => { for path in agent::list_files(&workspace)? { println!("{}", path.display()); } Ok(()) }
            AgentCommands::Test { workspace, command } => agent::run_test_command(&workspace, &command),
        },
        Commands::Status { checkpoint, corpus } => {
            println!("Clodex version {}", env!("CARGO_PKG_VERSION"));
            println!("Engine: {}", unsafe { CStr::from_ptr(clodex_engine_version()).to_string_lossy() });
            println!("Checkpoint: {} ({})", checkpoint.display(), if checkpoint.exists() { "present" } else { "missing" });
            println!("Corpus: {} ({})", corpus.display(), if corpus.is_file() { "present" } else { "missing" });
            if checkpoint.is_file() { println!("Checkpoint SHA-256: {}", hash_file(&checkpoint)?); }
            Ok(())
        }
        Commands::InspectCorpus { path } => {
            let bytes = fs::read(&path).with_context(|| format!("cannot read {}", path.display()))?;
            if bytes.is_empty() { bail!("corpus is empty"); }
            let text = String::from_utf8_lossy(&bytes);
            let lines = text.lines().count();
            let words = text.split_whitespace().count();
            println!("path: {}", path.display());
            println!("bytes: {}", bytes.len());
            println!("lines: {}", lines);
            println!("whitespace tokens (rough estimate): {}", words);
            println!("sha256: {}", hash_file(&path)?);
            println!("UTF-8 byte IDs (Rust tokenizer): {}", tokenizer::encode(&text, false).len());
            println!("UTF-8 byte tokenizer vocabulary: 258 IDs (BOS=256, EOS=257)");
            Ok(())
        }
    }
}
