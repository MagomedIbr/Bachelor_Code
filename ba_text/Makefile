all: 
	pdflatex thesis.tex
	makeindex thesis.idx
	pdflatex thesis.tex
	bibtex thesis
	pdflatex thesis.tex
	bibtex thesis
	pdflatex thesis.tex
	
	
clean:
	rm -f *~
	rm -f *.aux
	rm -f *.bbl
	rm -f *.idx
	rm -f *.ind
	rm -f *.toc
	rm -f *.out
	rm -f *.pdf
	rm -f *.log
	rm -f *.blg
	rm -f *.ilg
	rm -f *.backup
	rm -f *-blx.bib
	rm -f *.run.xml
	rm -f *.lof
	rm -f *.lot
