# !/bin/bash
# file based on answer:
# https://tex.stackexchange.com/questions/140845/how-can-i-ignore-latex-error-while-compiling

# options: [main, 1pagequant]
filename=$1
name=trung-vt

mv junk/${filename}.lo* junk/${filename}.aux junk/${filename}.ilg .
mv junk/${filename}.ind junk/${filename}.toc .
mv junk/${filename}.bbl junk/${filename}.blg .
mv junk/${filename}.out junk/${filename}.asc .
mv junk/${filename}.snm junk/${filename}.fls junk/${filename}.run.xml .
mv junk/${filename}.nav junk/${filename}.dvi junk/${filename}.fdb_latexmk .
mv junk/${filename}.vrb  junk/${filename}-blx.bib .


pdflatex --interaction nonstopmode --shell-escape ${filename}.tex # >/dev/null
makeindex -c -s myindex.ist ${filename}.idx #2>/dev/null
# bibtex ${filename} #>/dev/null
biber ${filename} #>/dev/null
makeindex -c -s myindex.ist ${filename}.idx #2>/dev/null
pdflatex  --interaction nonstopmode  --shell-escape ${filename}.tex #>/dev/null


mv ${filename}.lo* *.aux ${filename}.ilg ${filename}.ind ${filename}.toc junk/
mv ${filename}.bbl ${filename}.blg ${filename}.out *.asc junk/
mv *.snm *.fls *.run.xml *.nav *.dvi *.fdb_latexmk *.vrb junk/
mv *-blx.bib junk/
# mv ${filename}.pdf ${name}-cv.pdf # rename to cv

