# Contributing to DermaFusion-AI

Thank you for your interest in contributing to DermaFusion-AI! This is a research-grade skin cancer detection project. We welcome bug reports, suggestions, and contributions from the community.

## Ways to Contribute

### 🐛 Bug Reports
If you find a bug, please open an [Issue](https://github.com/ai-with-abdullah/DermaFusion-AI/issues) with:
- A clear title describing the problem
- Steps to reproduce the issue
- Expected vs actual behaviour
- Your environment (OS, Python version, GPU, etc.)

### 💡 Feature Suggestions
Open an Issue with the label `enhancement` describing:
- What you want to add and why
- Any relevant references (papers, tools, datasets)

### 🔧 Code Contributions
1. **Fork** the repository
2. **Create a branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes** and test them
4. **Commit** with a clear message: `git commit -m "feat: add your feature"`
5. **Push**: `git push origin feature/your-feature-name`
6. Open a **Pull Request** targeting the `main` branch

## Code Style

- Follow PEP 8 for Python code
- Add docstrings to all functions and classes
- Keep functions focused and under ~50 lines where possible
- Add comments for non-obvious logic

## Research Contributions

If you evaluate DermaFusion-AI on a new dataset or extend the architecture, please:
- Document your experimental setup clearly
- Report all standard metrics (AUC, Balanced Accuracy, MEL Sensitivity)
- Use patient-aware splitting if your dataset has multiple images per patient

## Questions

Open a GitHub Issue with the label `question` or contact the maintainer through the repository.

---

*This project targets submission to MDPI Diagnostics and MIDL 2026. Contributions that improve cross-domain generalisation, model efficiency, or clinical explainability are especially welcome.*
