# muvr MLP analysis package

This package provides analysis capabilities using Multilayer Perceptrons. It is written in python using spark and neon.

## Installation & Running the application

Make sure you have the package `python-setuptools` installed (eg. using `pip install setuptools`). After that, install using

```bash
python setup.py install
```

Run the analysis using

```bash
python start_analysis.py
```

or using

```bash
python start_training.py -m core -d ../../muvr-training-data/labelled/core -o ../output/ -v ../output/v.png -e  ../output/e.csv
```
