package promptabi

import (
	"os"
	"path/filepath"
	"testing"
)

func TestReadsGeneratedPromptABIFixtures(t *testing.T) {
	fixtures := os.Getenv("PROMPTABI_SDK_FIXTURES")
	if fixtures == "" {
		t.Skip("PROMPTABI_SDK_FIXTURES is not set")
	}

	report, err := ReadIntegrationReport(filepath.Join(fixtures, "integration-report.json"))
	if err != nil {
		t.Fatal(err)
	}
	if report.Protocol != IntegrationProtocol || SummarizeReport(report).DiagnosticCount == 0 {
		t.Fatalf("unexpected integration report summary: %#v", SummarizeReport(report))
	}

	lockfile, err := ReadLockfile(filepath.Join(fixtures, "lockfile.json"))
	if err != nil {
		t.Fatal(err)
	}
	if lockfile.LockfileVersion != LockfileVersion || len(lockfile.Checks) == 0 {
		t.Fatalf("unexpected lockfile: %#v", lockfile)
	}

	bundle, err := ReadVerificationBundle(filepath.Join(fixtures, "bundle.json"))
	if err != nil {
		t.Fatal(err)
	}
	diagnostics := DiagnosticsFromBundle(bundle)
	if bundle.Payload.BundleVersion != BundleVersion || len(diagnostics) == 0 {
		t.Fatalf("unexpected bundle diagnostics: %#v", diagnostics)
	}

	payload, err := ReadDiagnosticPayload(filepath.Join(fixtures, "diagnostics.json"))
	if err != nil {
		t.Fatal(err)
	}
	summary := SummarizeDiagnostics(payload.Diagnostics, payload.OK != nil && *payload.OK)
	if summary.DiagnosticCount == 0 || len(summary.Fingerprints) == 0 {
		t.Fatalf("unexpected diagnostic payload summary: %#v", summary)
	}
}
