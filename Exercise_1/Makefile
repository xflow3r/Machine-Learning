# Makefile for LaTeX report in Exercise_0


# Main LaTeX source file (without extension)
REPORT=Group68_Campbell_Kohl_Scherling

# All file extensions to clean (from .gitignore)
LATEX_EXTS=aux log out toc bbl blg synctex.gz fdb_latexmk fls lof lot idx ind ilg nav snm vrb pdf

clean:
	@echo "Cleaning generated LaTeX files..."
	@for ext in $(LATEX_EXTS); do \
		rm -f $(REPORT).$$ext; \
	done
	@echo "Done."
