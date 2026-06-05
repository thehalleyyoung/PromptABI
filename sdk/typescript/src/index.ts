import { readFileSync } from "node:fs";

export const PROMPTABI_INTEGRATION_PROTOCOL = "promptabi.integration.v1";
export const PROMPTABI_LOCKFILE_VERSION = 1;
export const PROMPTABI_BUNDLE_VERSION = 1;

export type JsonObject = Record<string, unknown>;

export interface PromptAbiDiagnostic {
  rule_id: string;
  severity: string;
  message: string;
  fingerprint?: string;
  artifact?: JsonObject | null;
  span?: JsonObject | null;
  witness?: JsonObject | null;
  properties?: unknown;
}

export interface PromptAbiIntegrationReport {
  protocol: typeof PROMPTABI_INTEGRATION_PROTOCOL;
  request: JsonObject;
  gate: "pass" | "warn" | "fail";
  ok: boolean;
  diagnostic_counts: Record<string, number>;
  artifacts: JsonObject[];
  capabilities: JsonObject[];
  surfaces: Record<string, unknown>;
}

export interface PromptAbiLockfile {
  lockfile_version: typeof PROMPTABI_LOCKFILE_VERSION;
  promptabi_version: string;
  config_name: string;
  config_hash: string;
  artifacts: JsonObject[];
  checks: string[];
  diagnostic_baseline: JsonObject[];
  library_versions: Record<string, string>;
  provider_fixture_versions?: Record<string, string>;
}

export interface PromptAbiVerificationBundle {
  algorithm: string;
  bundle_hash: string;
  payload: {
    bundle_version: typeof PROMPTABI_BUNDLE_VERSION;
    diagnostics: PromptAbiDiagnostic[];
    lockfile: PromptAbiLockfile;
    reproducibility: JsonObject;
    [key: string]: unknown;
  };
  signature: string;
  signing_key_id: string;
}

export interface PromptAbiDiagnosticPayload {
  diagnostics: PromptAbiDiagnostic[];
  ok?: boolean;
  [key: string]: unknown;
}

export interface PromptAbiSummary {
  ok: boolean;
  gate: "pass" | "warn" | "fail";
  diagnosticCount: number;
  errorCount: number;
  warningCount: number;
  fingerprints: string[];
}

export function readJsonFile(path: string): unknown {
  return JSON.parse(readFileSync(path, "utf8"));
}

export function readIntegrationReport(path: string): PromptAbiIntegrationReport {
  const value = expectObject(readJsonFile(path), "integration report");
  if (value.protocol !== PROMPTABI_INTEGRATION_PROTOCOL) {
    throw new Error(`unsupported PromptABI integration protocol: ${String(value.protocol)}`);
  }
  return value as unknown as PromptAbiIntegrationReport;
}

export function readLockfile(path: string): PromptAbiLockfile {
  const value = expectObject(readJsonFile(path), "lockfile");
  if (value.lockfile_version !== PROMPTABI_LOCKFILE_VERSION) {
    throw new Error(`unsupported PromptABI lockfile version: ${String(value.lockfile_version)}`);
  }
  return value as unknown as PromptAbiLockfile;
}

export function readVerificationBundle(path: string): PromptAbiVerificationBundle {
  const value = expectObject(readJsonFile(path), "verification bundle");
  const payload = expectObject(value.payload, "verification bundle payload");
  if (payload.bundle_version !== PROMPTABI_BUNDLE_VERSION) {
    throw new Error(`unsupported PromptABI bundle version: ${String(payload.bundle_version)}`);
  }
  return value as unknown as PromptAbiVerificationBundle;
}

export function readDiagnosticPayload(path: string): PromptAbiDiagnosticPayload {
  const value = expectObject(readJsonFile(path), "diagnostic payload");
  if (!Array.isArray(value.diagnostics)) {
    throw new Error("PromptABI diagnostic payload must contain a diagnostics array");
  }
  return value as unknown as PromptAbiDiagnosticPayload;
}

export function diagnosticsFromBundle(bundle: PromptAbiVerificationBundle): PromptAbiDiagnostic[] {
  return bundle.payload.diagnostics;
}

export function summarizeDiagnostics(diagnostics: PromptAbiDiagnostic[], ok = false): PromptAbiSummary {
  const errorCount = diagnostics.filter((diagnostic) => diagnostic.severity === "error").length;
  const warningCount = diagnostics.filter((diagnostic) => diagnostic.severity === "warning").length;
  return {
    ok,
    gate: errorCount > 0 ? "fail" : warningCount > 0 ? "warn" : "pass",
    diagnosticCount: diagnostics.length,
    errorCount,
    warningCount,
    fingerprints: diagnostics
      .map((diagnostic) => diagnostic.fingerprint)
      .filter((fingerprint): fingerprint is string => typeof fingerprint === "string"),
  };
}

export function summarizeReport(report: PromptAbiIntegrationReport): PromptAbiSummary {
  const errorCount = report.diagnostic_counts.error ?? 0;
  const warningCount = report.diagnostic_counts.warning ?? 0;
  return {
    ok: report.ok,
    gate: report.gate,
    diagnosticCount: Object.values(report.diagnostic_counts).reduce((total, count) => total + count, 0),
    errorCount,
    warningCount,
    fingerprints: [],
  };
}

function expectObject(value: unknown, description: string): JsonObject {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`PromptABI ${description} must be a JSON object`);
  }
  return value as JsonObject;
}
