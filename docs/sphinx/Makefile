# Makefile for Sphinx documentation

# You can set these variables from the command line
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help"
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Custom targets
.PHONY: html clean rebuild apidoc serve

html:
	@echo "Building HTML documentation..."
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILDDIR)
	@echo "Clean complete."

rebuild: clean html
	@echo "Rebuild complete."

apidoc:
	@echo "Regenerating API documentation files..."
	@sphinx-apidoc -f -o . ../../models --separate
	@sphinx-apidoc -f -o . ../../utils --separate
	@echo "API documentation regenerated."

serve: html
	@echo "Starting local documentation server at http://localhost:8000"
	@echo "Press Ctrl+C to stop."
	@cd $(BUILDDIR)/html && python3 -m http.server 8000

update: apidoc html
	@echo "Documentation updated."
