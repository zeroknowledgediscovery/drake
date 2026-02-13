ALL_FIGURE_NAMES=$(shell cat main.figlist)
ALL_FIGURES=$(ALL_FIGURE_NAMES:%=%.pdf)

allimages: $(ALL_FIGURES)
	@echo All images exist now. Use make -B to re-generate them.

FORCEREMAKE:

-include $(ALL_FIGURE_NAMES:%=%.dep)

%.dep:
	mkdir -p "$(dir $@)"
	touch "$@" # will be filled later.

./Figures/External/figperf.pdf: 
	pdflatex -shell-escape -halt-on-error -interaction=batchmode -jobname "./Figures/External/figperf" "\def\tikzexternalrealjob{main}\input{main}"

./Figures/External/figperf.pdf: ./Figures/External/figperf.md5
