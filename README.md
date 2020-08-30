# The Sinkhorn's Algorithm from Scratch

This repository implements **Sinkhorn's Algorithm**, an algorithm to solve discrete optimal transport problem.

You can move source and target points and change the regularization weight _gamma_ interactively.

![ot_demo](https://user-images.githubusercontent.com/25974220/91658862-dee37080-eb06-11ea-8a57-91ee60f1c9da.gif)

## Install
Poetry or pip
- `poetry install`
- `pip install -r requirements.txt`


## Requirements

- Python 3.7+
- numpy
- scipy
- cython
- [POT](https://pythonot.github.io/)

## Reference

- [Optimal Transport II (Cuturi, MLSS2020)](http://mlss.tuebingen.mpg.de/2020/index.html)
