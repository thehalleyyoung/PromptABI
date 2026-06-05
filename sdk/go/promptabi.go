package promptabi

import (
	"encoding/json"
	"fmt"
	"os"
)

const IntegrationProtocol = "promptabi.integration.v1"
const LockfileVersion = 1
const BundleVersion = 1

type Object = map[string]any

type Diagnostic struct {
	RuleID      string `json:"rule_id"`
	Severity    string `json:"severity"`
	Message     string `json:"message"`
	Fingerprint string `json:"fingerprint,omitempty"`
	Artifact    Object `json:"artifact,omitempty"`
	Span        Object `json:"span,omitempty"`
	Witness     Object `json:"witness,omitempty"`
}

type IntegrationReport struct {
	Protocol         string         `json:"protocol"`
	Request          Object         `json:"request"`
	Gate             string         `json:"gate"`
	OK               bool           `json:"ok"`
	DiagnosticCounts map[string]int `json:"diagnostic_counts"`
	Artifacts        []Object       `json:"artifacts"`
	Capabilities     []Object       `json:"capabilities"`
	Surfaces         Object         `json:"surfaces"`
}

type Lockfile struct {
	LockfileVersion        int               `json:"lockfile_version"`
	PromptABIVersion       string            `json:"promptabi_version"`
	ConfigName             string            `json:"config_name"`
	ConfigHash             string            `json:"config_hash"`
	Artifacts              []Object          `json:"artifacts"`
	Checks                 []string          `json:"checks"`
	DiagnosticBaseline     []Object          `json:"diagnostic_baseline"`
	LibraryVersions        map[string]string `json:"library_versions"`
	ProviderFixtureVersion map[string]string `json:"provider_fixture_versions,omitempty"`
}

type VerificationBundle struct {
	Algorithm    string        `json:"algorithm"`
	BundleHash   string        `json:"bundle_hash"`
	Payload      BundlePayload `json:"payload"`
	Signature    string        `json:"signature"`
	SigningKeyID string        `json:"signing_key_id"`
}

type BundlePayload struct {
	BundleVersion   int          `json:"bundle_version"`
	Diagnostics     []Diagnostic `json:"diagnostics"`
	Lockfile        Lockfile     `json:"lockfile"`
	Reproducibility Object       `json:"reproducibility"`
}

type DiagnosticPayload struct {
	Diagnostics []Diagnostic `json:"diagnostics"`
	OK          *bool        `json:"ok,omitempty"`
}

type Summary struct {
	OK              bool
	Gate            string
	DiagnosticCount int
	ErrorCount      int
	WarningCount    int
	Fingerprints    []string
}

func ReadIntegrationReport(path string) (*IntegrationReport, error) {
	var report IntegrationReport
	if err := readJSON(path, &report); err != nil {
		return nil, err
	}
	if report.Protocol != IntegrationProtocol {
		return nil, fmt.Errorf("unsupported PromptABI integration protocol: %q", report.Protocol)
	}
	return &report, nil
}

func ReadLockfile(path string) (*Lockfile, error) {
	var lockfile Lockfile
	if err := readJSON(path, &lockfile); err != nil {
		return nil, err
	}
	if lockfile.LockfileVersion != LockfileVersion {
		return nil, fmt.Errorf("unsupported PromptABI lockfile version: %d", lockfile.LockfileVersion)
	}
	return &lockfile, nil
}

func ReadVerificationBundle(path string) (*VerificationBundle, error) {
	var bundle VerificationBundle
	if err := readJSON(path, &bundle); err != nil {
		return nil, err
	}
	if bundle.Payload.BundleVersion != BundleVersion {
		return nil, fmt.Errorf("unsupported PromptABI bundle version: %d", bundle.Payload.BundleVersion)
	}
	return &bundle, nil
}

func ReadDiagnosticPayload(path string) (*DiagnosticPayload, error) {
	var payload DiagnosticPayload
	if err := readJSON(path, &payload); err != nil {
		return nil, err
	}
	if payload.Diagnostics == nil {
		return nil, fmt.Errorf("PromptABI diagnostic payload must contain diagnostics")
	}
	return &payload, nil
}

func DiagnosticsFromBundle(bundle *VerificationBundle) []Diagnostic {
	return bundle.Payload.Diagnostics
}

func SummarizeDiagnostics(diagnostics []Diagnostic, ok bool) Summary {
	summary := Summary{OK: ok, Gate: "pass", DiagnosticCount: len(diagnostics)}
	for _, diagnostic := range diagnostics {
		if diagnostic.Fingerprint != "" {
			summary.Fingerprints = append(summary.Fingerprints, diagnostic.Fingerprint)
		}
		switch diagnostic.Severity {
		case "error":
			summary.ErrorCount++
		case "warning":
			summary.WarningCount++
		}
	}
	if summary.ErrorCount > 0 {
		summary.Gate = "fail"
	} else if summary.WarningCount > 0 {
		summary.Gate = "warn"
	}
	return summary
}

func SummarizeReport(report *IntegrationReport) Summary {
	summary := Summary{OK: report.OK, Gate: report.Gate}
	for severity, count := range report.DiagnosticCounts {
		summary.DiagnosticCount += count
		if severity == "error" {
			summary.ErrorCount = count
		}
		if severity == "warning" {
			summary.WarningCount = count
		}
	}
	return summary
}

func readJSON(path string, target any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(data, target); err != nil {
		return err
	}
	return nil
}
