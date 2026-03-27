---
description: "Use when running any terminal commands, scripts, or Python code in this project. Always activate the conda environment 'llm' before executing."
applyTo: "**"
---
# Conda Environment

Always activate the `llm` conda environment before running any terminal command in this project:

```bash
conda activate llm && <your command>
```

- Use `conda activate llm &&` as a prefix for all terminal commands, including `python`, `pytest`, `pip`, `git`, `ls`, etc.
- Never assume the environment is already active.
- If installing packages, always run inside the `llm` environment: `conda activate llm && pip install ...`
