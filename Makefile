# Simple helpers
.PHONY: data analysis report

data:
	python src/data_download.py

analysis:
	python src/q1_binomial.py
	python src/q2_pension.py
	python src/q3_portfolio.py
	python src/q4_forwards.py

report:
	quarto render hw1.qmd
