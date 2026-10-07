# Source and build
Repository: https://github.com/thehalleyyoung/PromptABI (this revision is local until owner integration).
Canonical manuscript source: paper/tool_paper.tex, paper/numbers.tex, paper/references.bib. Run ./paper/build.sh with latexmk, pdflatex and IEEEtran installed; output is root tool_paper.pdf. The baseline source and hash are in versions/.
Synthetic probe: Python >=3.11, jinja2 and tokenizers. Run PYTHONPATH=src python3 study/wave2_probe.py. No model weights/network required. Environment versions and exact frozen-case hash are in before.json and after.json. Static+prior focused regression command is in checks.json and tests.log.
Pages: python scripts/build_site.py, then serve _site locally. Workflow is manual and prepared only; no push, deployment or upload occurred.
