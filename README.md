
## Data
Folder `data/` contains all datas retrieve from the Swiss Federal Statistical Office (FSO).

Folder `result/` contains all the results of the analysis and the images used in the report.

Folder `venv/` contains the virtual environment used for this project.

.py files are the scripts used to perform the analysis.

metadata.yaml contains the metadata for the report generation.

report.md is the markdown file containing the report.

## Create the Pdf with pandoc on mac

1. Install https://www.tug.org/mactex/mactex-download.html

2. Run pandoc to generate the pdf:

``` bash
pandoc report.md \
  --metadata-file=metadata.yaml \
  --pdf-engine=xelatex \
  -o build/report.pdf
```

3. The PDF is build on `build/report.pdf`