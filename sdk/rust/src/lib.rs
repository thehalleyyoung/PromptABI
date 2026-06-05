use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

pub const PROMPTABI_INTEGRATION_PROTOCOL: &str = "promptabi.integration.v1";
pub const PROMPTABI_LOCKFILE_VERSION: u64 = 1;
pub const PROMPTABI_BUNDLE_VERSION: u64 = 1;

pub type Object = BTreeMap<String, Value>;

#[derive(Debug, Deserialize)]
pub struct Diagnostic {
    pub rule_id: String,
    pub severity: String,
    pub message: String,
    #[serde(default)]
    pub fingerprint: Option<String>,
    #[serde(default)]
    pub artifact: Option<Object>,
    #[serde(default)]
    pub span: Option<Object>,
    #[serde(default)]
    pub witness: Option<Object>,
}

#[derive(Debug, Deserialize)]
pub struct IntegrationReport {
    pub protocol: String,
    pub request: Object,
    pub gate: String,
    pub ok: bool,
    pub diagnostic_counts: BTreeMap<String, u64>,
    pub artifacts: Vec<Object>,
    pub capabilities: Vec<Object>,
    pub surfaces: Object,
}

#[derive(Debug, Deserialize)]
pub struct Lockfile {
    pub lockfile_version: u64,
    pub promptabi_version: String,
    pub config_name: String,
    pub config_hash: String,
    pub artifacts: Vec<Object>,
    pub checks: Vec<String>,
    pub diagnostic_baseline: Vec<Object>,
    pub library_versions: BTreeMap<String, String>,
    #[serde(default)]
    pub provider_fixture_versions: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
pub struct VerificationBundle {
    pub algorithm: String,
    pub bundle_hash: String,
    pub payload: BundlePayload,
    pub signature: String,
    pub signing_key_id: String,
}

#[derive(Debug, Deserialize)]
pub struct BundlePayload {
    pub bundle_version: u64,
    pub diagnostics: Vec<Diagnostic>,
    pub lockfile: Lockfile,
    pub reproducibility: Object,
}

#[derive(Debug, Deserialize)]
pub struct DiagnosticPayload {
    pub diagnostics: Vec<Diagnostic>,
    #[serde(default)]
    pub ok: Option<bool>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Summary {
    pub ok: bool,
    pub gate: String,
    pub diagnostic_count: usize,
    pub error_count: usize,
    pub warning_count: usize,
    pub fingerprints: Vec<String>,
}

pub fn read_integration_report(path: impl AsRef<Path>) -> Result<IntegrationReport, String> {
    let report: IntegrationReport = read_json(path)?;
    if report.protocol != PROMPTABI_INTEGRATION_PROTOCOL {
        return Err(format!(
            "unsupported PromptABI integration protocol: {}",
            report.protocol
        ));
    }
    Ok(report)
}

pub fn read_lockfile(path: impl AsRef<Path>) -> Result<Lockfile, String> {
    let lockfile: Lockfile = read_json(path)?;
    if lockfile.lockfile_version != PROMPTABI_LOCKFILE_VERSION {
        return Err(format!(
            "unsupported PromptABI lockfile version: {}",
            lockfile.lockfile_version
        ));
    }
    Ok(lockfile)
}

pub fn read_verification_bundle(path: impl AsRef<Path>) -> Result<VerificationBundle, String> {
    let bundle: VerificationBundle = read_json(path)?;
    if bundle.payload.bundle_version != PROMPTABI_BUNDLE_VERSION {
        return Err(format!(
            "unsupported PromptABI bundle version: {}",
            bundle.payload.bundle_version
        ));
    }
    Ok(bundle)
}

pub fn read_diagnostic_payload(path: impl AsRef<Path>) -> Result<DiagnosticPayload, String> {
    read_json(path)
}

pub fn diagnostics_from_bundle(bundle: &VerificationBundle) -> &[Diagnostic] {
    &bundle.payload.diagnostics
}

pub fn summarize_diagnostics(diagnostics: &[Diagnostic], ok: bool) -> Summary {
    let error_count = diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.severity == "error")
        .count();
    let warning_count = diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.severity == "warning")
        .count();
    Summary {
        ok,
        gate: if error_count > 0 {
            "fail".to_string()
        } else if warning_count > 0 {
            "warn".to_string()
        } else {
            "pass".to_string()
        },
        diagnostic_count: diagnostics.len(),
        error_count,
        warning_count,
        fingerprints: diagnostics
            .iter()
            .filter_map(|diagnostic| diagnostic.fingerprint.clone())
            .collect(),
    }
}

pub fn summarize_report(report: &IntegrationReport) -> Summary {
    Summary {
        ok: report.ok,
        gate: report.gate.clone(),
        diagnostic_count: report.diagnostic_counts.values().sum::<u64>() as usize,
        error_count: *report.diagnostic_counts.get("error").unwrap_or(&0) as usize,
        warning_count: *report.diagnostic_counts.get("warning").unwrap_or(&0) as usize,
        fingerprints: Vec::new(),
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(path: impl AsRef<Path>) -> Result<T, String> {
    let text = fs::read_to_string(path.as_ref())
        .map_err(|error| format!("failed to read {}: {error}", path.as_ref().display()))?;
    serde_json::from_str(&text)
        .map_err(|error| format!("failed to parse {}: {error}", path.as_ref().display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::path::PathBuf;

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env::var("PROMPTABI_SDK_FIXTURES").expect("PROMPTABI_SDK_FIXTURES is set"))
            .join(name)
    }

    #[test]
    fn reads_generated_promptabi_fixtures() {
        let Ok(_) = env::var("PROMPTABI_SDK_FIXTURES") else {
            return;
        };
        let report = read_integration_report(fixture("integration-report.json")).unwrap();
        assert_eq!(report.protocol, PROMPTABI_INTEGRATION_PROTOCOL);
        assert!(summarize_report(&report).diagnostic_count > 0);

        let lockfile = read_lockfile(fixture("lockfile.json")).unwrap();
        assert_eq!(lockfile.lockfile_version, PROMPTABI_LOCKFILE_VERSION);
        assert!(!lockfile.checks.is_empty());

        let bundle = read_verification_bundle(fixture("bundle.json")).unwrap();
        assert_eq!(bundle.payload.bundle_version, PROMPTABI_BUNDLE_VERSION);
        assert!(!diagnostics_from_bundle(&bundle).is_empty());

        let payload = read_diagnostic_payload(fixture("diagnostics.json")).unwrap();
        let summary = summarize_diagnostics(&payload.diagnostics, payload.ok.unwrap_or(false));
        assert!(summary.diagnostic_count > 0);
        assert!(!summary.fingerprints.is_empty());
    }
}
