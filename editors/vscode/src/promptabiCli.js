"use strict";

const path = require("path");

function buildDiagnosticsArgs(options = {}) {
  const args = ["diagnostics", "lsp", "--format", "json"];
  if (options.configPath) {
    args.push("--config", options.configPath);
  }
  if (options.workspaceRoot) {
    args.push("--workspace-root", options.workspaceRoot);
  }
  for (const artifact of options.artifacts || []) {
    if (artifact) {
      args.push("--artifact", artifact);
    }
  }
  for (const plugin of options.plugins || []) {
    if (plugin) {
      args.push("--plugin", plugin);
    }
  }
  return args;
}

function buildExplainArgs(options = {}) {
  const args = ["explain"];
  if (options.configPath) {
    args.push("--config", options.configPath);
  }
  if (options.fingerprint) {
    args.push("--fingerprint", options.fingerprint);
  } else if (options.ruleId) {
    args.push("--rule-id", options.ruleId);
  } else if (options.index) {
    args.push("--index", String(options.index));
  }
  return args;
}

function workspaceRootFromFolders(folders) {
  if (!folders || !folders.length) {
    return process.cwd();
  }
  const first = folders[0].uri && folders[0].uri.fsPath ? folders[0].uri.fsPath : String(folders[0]);
  return path.resolve(first);
}

function diagnosticDocuments(payload) {
  if (!payload || !Array.isArray(payload.documents)) {
    return [];
  }
  return payload.documents.filter((document) => document && document.params && document.params.uri);
}

function collectWitnessPreviews(payload) {
  const previews = [];
  for (const document of diagnosticDocuments(payload)) {
    const diagnostics = document.params.diagnostics || [];
    for (const diagnostic of diagnostics) {
      const data = diagnostic.data || {};
      const witness = data.witness;
      if (!witness || !Array.isArray(witness.steps) || !witness.steps.length) {
        continue;
      }
      const steps = witness.steps
        .map((step, index) => {
          const pieces = [`${index + 1}. ${step.action || "witness step"}`];
          if (step.input !== undefined) {
            pieces.push(`input: ${String(step.input)}`);
          }
          if (step.output !== undefined) {
            pieces.push(`output: ${String(step.output)}`);
          }
          return pieces.join(" | ");
        })
        .join("\n");
      previews.push({
        uri: document.params.uri,
        code: diagnostic.code,
        message: diagnostic.message,
        fingerprint: data.fingerprint,
        summary: witness.summary || diagnostic.message,
        steps
      });
    }
  }
  return previews;
}

function renderWitnessMarkdown(previews) {
  if (!previews.length) {
    return "# PromptABI witness preview\n\nNo diagnostics with rendered/tokenized witnesses are available.";
  }
  const sections = previews.map((preview) => {
    return [
      `## ${preview.code}`,
      "",
      preview.message,
      "",
      `- Document: ${preview.uri}`,
      preview.fingerprint ? `- Fingerprint: ${preview.fingerprint}` : "",
      "",
      "```text",
      preview.steps,
      "```"
    ]
      .filter(Boolean)
      .join("\n");
  });
  return `# PromptABI witness preview\n\n${sections.join("\n\n")}\n`;
}

module.exports = {
  buildDiagnosticsArgs,
  buildExplainArgs,
  collectWitnessPreviews,
  diagnosticDocuments,
  renderWitnessMarkdown,
  workspaceRootFromFolders
};
