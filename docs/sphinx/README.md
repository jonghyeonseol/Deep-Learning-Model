# Sphinx API Documentation

This directory contains the Sphinx-generated API documentation for the Deep Learning Framework.

## Quick Start

**View the documentation:**

```bash
# Open the main page in your browser
open _build/html/index.html

# Or use a simple HTTP server
cd _build/html
python3 -m http.server 8000
# Then visit http://localhost:8000
```

## Building Documentation

**Requirements:**

```bash
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints tensorboard
```

**Build commands:**

```bash
# Build HTML documentation
sphinx-build -b html . _build/html

# Clean build artifacts
rm -rf _build

# Rebuild from scratch
rm -rf _build && sphinx-build -b html . _build/html

# Build with warnings as errors (strict mode)
sphinx-build -W -b html . _build/html
```

## Directory Structure

```
sphinx/
├── conf.py              # Sphinx configuration
├── index.rst            # Main documentation page
├── models.*.rst         # Auto-generated module docs (models package)
├── utils.*.rst          # Auto-generated module docs (utils package)
├── _build/              # Generated documentation (HTML, etc.)
│   └── html/            # HTML output
├── _static/             # Static files (CSS, JS, images)
└── _templates/          # Custom templates
```

## Updating Documentation

**After modifying docstrings:**

```bash
# Regenerate .rst files for models
sphinx-apidoc -f -o . ../../models --separate

# Regenerate .rst files for utils
sphinx-apidoc -f -o . ../../utils --separate

# Rebuild HTML
sphinx-build -b html . _build/html
```

**After adding new modules:**

1. Run sphinx-apidoc to generate .rst files
2. Add module references to index.rst if needed
3. Rebuild documentation

## Configuration

**conf.py highlights:**

- **Theme**: Read the Docs theme (sphinx_rtd_theme)
- **Extensions**:
  - `sphinx.ext.autodoc` - Auto-generate docs from docstrings
  - `sphinx.ext.viewcode` - Link to source code
  - `sphinx.ext.napoleon` - Google/NumPy style docstrings
  - `sphinx_autodoc_typehints` - Type hints support
  - `sphinx.ext.intersphinx` - Link to Python/PyTorch/NumPy docs
- **Autodoc settings**: Include special methods, sort by source order
- **Path**: Configured to find modules in `../../` (project root)

## Documentation Style

The framework uses **Google-style docstrings**:

```python
def train(epochs: int, learning_rate: float) -> dict:
    """Train the model.

    Args:
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer

    Returns:
        Dictionary with training history and metrics

    Raises:
        TrainingError: If training fails
        ValueError: If epochs <= 0

    Examples:
        >>> trainer.train(epochs=10, learning_rate=0.001)
        {'loss': [...], 'accuracy': [...]}
    """
    pass
```

## Hosting Documentation

**Local hosting:**

```bash
cd _build/html
python3 -m http.server 8000
```

**Read the Docs (recommended):**

1. Push documentation to GitHub
2. Connect repository to https://readthedocs.org
3. Configure build settings (automatically uses conf.py)
4. Documentation auto-builds on each commit

**GitHub Pages:**

```bash
# Build to docs/ directory in repo root
sphinx-build -b html . ../../../docs

# Enable GitHub Pages on the docs/ folder in settings
# Visit: https://username.github.io/repo-name/
```

## Troubleshooting

**Module import errors:**

- Ensure all dependencies are installed
- Check sys.path configuration in conf.py
- Verify package structure has `__init__.py` files

**Missing dependencies:**

```bash
pip install tensorboard  # Required for utils.trainer
pip install torch torchvision  # Required for models
```

**Warnings about missing files:**

- If you see warnings about `getting_started.rst` or similar, these are placeholder references
- Either create the files or remove references from index.rst

**Outdated documentation:**

```bash
# Clean and rebuild
rm -rf _build _modules
sphinx-build -b html . _build/html
```

## Advanced Features

**PDF Documentation:**

```bash
pip install latexmk
sphinx-build -b latex . _build/latex
cd _build/latex && make
```

**EPUB Documentation:**

```bash
sphinx-build -b epub . _build/epub
```

**Linkcheck (verify external links):**

```bash
sphinx-build -b linkcheck . _build/linkcheck
```

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [Read the Docs Theme](https://sphinx-rtd-theme.readthedocs.io/)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [Google Style Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
