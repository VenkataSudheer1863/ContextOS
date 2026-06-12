# =============================================================================
# ContextOS Paper -- LaTeX Build System
# =============================================================================
# Usage:
#   make          -- build PDF (default target)
#   make pdf      -- build PDF from contextos_main.tex
#   make clean    -- remove all LaTeX auxiliary files
#   make cleanall -- remove auxiliary files AND output PDF
#   make view     -- build PDF and open in default viewer
#   make watch    -- rebuild on source file changes (requires inotifywait)
#   make spell    -- run aspell spell-check on main tex file
#   make count    -- report approximate word count
#
# Requirements:
#   - pdflatex (TeX Live or MiKTeX)
#   - bibtex
#   - aspell (optional, for spell-check target)
#   - inotifywait (optional, from inotify-tools, for watch target)
# =============================================================================

# --- Configuration -----------------------------------------------------------

MAIN     := contextos_main
APPENDIX := appendix
BIB      := references
LATEX    := pdflatex
BIBTEX   := bibtex
LATEXFLAGS := -interaction=nonstopmode -halt-on-error
PDF      := $(MAIN).pdf

# Auxiliary file extensions produced by pdflatex/bibtex
AUX_EXTS := aux bbl blg log out toc lof lot synctex.gz fls fdb_latexmk \
            idx ind ilg nav snm vrb bcf run.xml

# All .tex source files (used for watch target dependency)
TEX_SRCS := $(wildcard *.tex)

# -----------------------------------------------------------------------------
# Default target
# -----------------------------------------------------------------------------

.DEFAULT_GOAL := pdf

.PHONY: all pdf clean cleanall view watch spell count help

all: pdf

# -----------------------------------------------------------------------------
# Build PDF
# -----------------------------------------------------------------------------

pdf: $(PDF)

$(PDF): $(MAIN).tex $(BIB).bib $(APPENDIX).tex
	@echo "==> Pass 1: pdflatex $(MAIN).tex"
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	@echo "==> Running bibtex $(MAIN)"
	$(BIBTEX) $(MAIN)
	@echo "==> Pass 2: pdflatex $(MAIN).tex (resolve citations)"
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	@echo "==> Pass 3: pdflatex $(MAIN).tex (resolve cross-references)"
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	@echo "==> Build complete: $(PDF)"

# -----------------------------------------------------------------------------
# Clean auxiliary files
# -----------------------------------------------------------------------------

clean:
	@echo "==> Removing LaTeX auxiliary files..."
	@rm -f $(foreach ext,$(AUX_EXTS),$(MAIN).$(ext))
	@rm -f $(foreach ext,$(AUX_EXTS),$(APPENDIX).$(ext))
	@echo "==> Clean complete."

# Clean auxiliary files AND the output PDF
cleanall: clean
	@echo "==> Removing output PDF..."
	@rm -f $(PDF)
	@echo "==> Cleanall complete."

# -----------------------------------------------------------------------------
# View PDF (platform-aware)
# -----------------------------------------------------------------------------

view: pdf
	@if command -v xdg-open >/dev/null 2>&1; then \
		xdg-open $(PDF); \
	elif command -v open >/dev/null 2>&1; then \
		open $(PDF); \
	elif command -v start >/dev/null 2>&1; then \
		start $(PDF); \
	else \
		echo "Cannot auto-open PDF; please open $(PDF) manually."; \
	fi

# -----------------------------------------------------------------------------
# Watch mode (auto-rebuild on source changes)
# Requires inotifywait: sudo apt install inotify-tools
# -----------------------------------------------------------------------------

watch:
	@echo "==> Watching for changes in *.tex and $(BIB).bib ..."
	@while true; do \
		inotifywait -q -e modify $(TEX_SRCS) $(BIB).bib; \
		echo "==> Change detected, rebuilding..."; \
		$(MAKE) pdf; \
	done

# -----------------------------------------------------------------------------
# Spell check (requires aspell)
# -----------------------------------------------------------------------------

spell:
	@echo "==> Running aspell on $(MAIN).tex ..."
	@aspell --mode=tex --lang=en_US check $(MAIN).tex
	@echo "==> Running aspell on $(APPENDIX).tex ..."
	@aspell --mode=tex --lang=en_US check $(APPENDIX).tex

# -----------------------------------------------------------------------------
# Word count estimate
# -----------------------------------------------------------------------------

count:
	@echo "==> Word count (texcount, approximate):"
	@if command -v texcount >/dev/null 2>&1; then \
		texcount -inc -total $(MAIN).tex; \
	else \
		echo "texcount not found; falling back to wc -w on detexed sources"; \
		detex $(MAIN).tex | wc -w; \
	fi

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------

help:
	@echo "ContextOS LaTeX Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make / make pdf   Build the PDF (3-pass pdflatex + bibtex)"
	@echo "  make clean        Remove LaTeX auxiliary files"
	@echo "  make cleanall     Remove auxiliary files AND the output PDF"
	@echo "  make view         Build PDF and open in default viewer"
	@echo "  make watch        Auto-rebuild on source changes (needs inotifywait)"
	@echo "  make spell        Run aspell spell-check (needs aspell)"
	@echo "  make count        Report approximate word count (needs texcount)"
	@echo "  make help         Show this help message"
	@echo ""
	@echo "Main source: $(MAIN).tex"
	@echo "Appendix:    $(APPENDIX).tex"
	@echo "Bibliography: $(BIB).bib"
	@echo "Output:      $(PDF)"
