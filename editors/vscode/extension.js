"use strict";

const cp = require("child_process");
const vscode = require("vscode");
const {
  buildDiagnosticsArgs,
  buildExplainArgs,
  collectWitnessPreviews,
  renderWitnessMarkdown,
  workspaceRootFromFolders
} = require("./src/promptabiCli");

let lastPayload = null;

function activate(context) {
  const collection = vscode.languages.createDiagnosticCollection("PromptABI");
  context.subscriptions.push(collection);

  const runQuickCheck = async () => {
    const payload = await runDiagnostics(collection);
    const count = payload.documents.reduce(
      (total, document) => total + (document.params.diagnostics || []).length,
      0
    );
    vscode.window.setStatusBarMessage(`PromptABI: ${count} diagnostic(s)`, 5000);
    return payload;
  };

  context.subscriptions.push(
    vscode.commands.registerCommand("promptabi.quickCheck", runQuickCheck),
    vscode.commands.registerCommand("promptabi.explainDiagnostic", () => explainSelectedDiagnostic(collection)),
    vscode.commands.registerCommand("promptabi.previewWitness", () => previewWitnesses()),
    vscode.workspace.onDidSaveTextDocument((document) => {
      if (shouldRunOnSave(document)) {
        runDiagnostics(collection).catch((error) => showPromptAbiError(error));
      }
    }),
    vscode.languages.registerCodeActionsProvider({ scheme: "file" }, new PromptAbiCodeActionProvider(), {
      providedCodeActionKinds: [vscode.CodeActionKind.QuickFix]
    })
  );

  runDiagnostics(collection).catch(() => undefined);
}

function deactivate() {}

async function runDiagnostics(collection) {
  const workspaceRoot = workspaceRootFromFolders(vscode.workspace.workspaceFolders);
  const config = vscode.workspace.getConfiguration("promptabi");
  const executable = config.get("executable") || "promptabi";
  const configPath = config.get("configPath") || undefined;
  const artifacts = config.get("extraArtifacts") || [];
  const args = buildDiagnosticsArgs({ configPath, workspaceRoot, artifacts });
  const payload = JSON.parse(await execPromptAbi(executable, args, workspaceRoot));
  lastPayload = payload;
  publishDiagnostics(collection, payload);
  return payload;
}

function publishDiagnostics(collection, payload) {
  collection.clear();
  for (const document of payload.documents || []) {
    const uri = vscode.Uri.parse(document.params.uri);
    const diagnostics = (document.params.diagnostics || []).map((item) => {
      const diagnostic = new vscode.Diagnostic(
        toRange(item.range),
        item.message,
        toSeverity(item.severity)
      );
      diagnostic.source = item.source || "PromptABI";
      diagnostic.code = item.code;
      diagnostic.promptabiData = item.data || {};
      diagnostic.relatedInformation = (item.relatedInformation || []).map((related) => {
        return new vscode.DiagnosticRelatedInformation(
          new vscode.Location(vscode.Uri.parse(related.location.uri), toRange(related.location.range)),
          related.message
        );
      });
      return diagnostic;
    });
    collection.set(uri, diagnostics);
  }
}

async function explainSelectedDiagnostic(collection) {
  const editor = vscode.window.activeTextEditor;
  const workspaceRoot = workspaceRootFromFolders(vscode.workspace.workspaceFolders);
  const config = vscode.workspace.getConfiguration("promptabi");
  const executable = config.get("executable") || "promptabi";
  const diagnostics = editor ? collection.get(editor.document.uri) || [] : [];
  const selected = diagnostics.find((diagnostic) => diagnostic.promptabiData);
  const data = selected ? selected.promptabiData : {};
  const args = buildExplainArgs({
    configPath: config.get("configPath") || undefined,
    fingerprint: data.fingerprint,
    ruleId: selected && selected.code ? String(selected.code) : undefined
  });
  const explanation = await execPromptAbi(executable, args, workspaceRoot);
  const doc = await vscode.workspace.openTextDocument({
    content: explanation,
    language: "markdown"
  });
  await vscode.window.showTextDocument(doc, { preview: true });
}

async function previewWitnesses() {
  if (!lastPayload) {
    await vscode.commands.executeCommand("promptabi.quickCheck");
  }
  const markdown = renderWitnessMarkdown(collectWitnessPreviews(lastPayload));
  const doc = await vscode.workspace.openTextDocument({
    content: markdown,
    language: "markdown"
  });
  await vscode.window.showTextDocument(doc, { preview: true });
}

function shouldRunOnSave(document) {
  const config = vscode.workspace.getConfiguration("promptabi");
  if (!config.get("runOnSave")) {
    return false;
  }
  return /(\.promptabi\.json|promptabi\.json|tokenizer_config\.json|schema\.json|tools?\.json)$/i.test(
    document.fileName
  );
}

function execPromptAbi(executable, args, cwd) {
  return new Promise((resolve, reject) => {
    cp.execFile(executable, args, { cwd, maxBuffer: 10 * 1024 * 1024 }, (error, stdout, stderr) => {
      if (stdout) {
        resolve(stdout);
        return;
      }
      if (error) {
        reject(new Error(stderr || error.message));
        return;
      }
      resolve("");
    });
  });
}

function toRange(range) {
  return new vscode.Range(
    new vscode.Position(range.start.line, range.start.character),
    new vscode.Position(range.end.line, range.end.character)
  );
}

function toSeverity(severity) {
  return {
    1: vscode.DiagnosticSeverity.Error,
    2: vscode.DiagnosticSeverity.Warning,
    3: vscode.DiagnosticSeverity.Information,
    4: vscode.DiagnosticSeverity.Hint
  }[severity] || vscode.DiagnosticSeverity.Information;
}

function showPromptAbiError(error) {
  vscode.window.showErrorMessage(`PromptABI: ${error.message || error}`);
}

class PromptAbiCodeActionProvider {
  provideCodeActions(document, range, context) {
    return context.diagnostics
      .filter((diagnostic) => diagnostic.source === "PromptABI")
      .map((diagnostic) => {
        const action = new vscode.CodeAction(
          `Explain PromptABI finding: ${diagnostic.code}`,
          vscode.CodeActionKind.QuickFix
        );
        action.command = {
          command: "promptabi.explainDiagnostic",
          title: "Explain PromptABI finding"
        };
        action.diagnostics = [diagnostic];
        return action;
      });
  }
}

module.exports = {
  activate,
  deactivate
};
