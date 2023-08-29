# BENK method
This repository contains the code base for ["The Beran Estimator with Neural Kernels for Estimating the Heterogeneous Treatment Effect"](https://arxiv.org/abs/2211.10793) article. The program is written in python 3 using the Pytorch framework.

To satisfy the dependencies use pip and requirements.txt:

```python
pip install -r requirements.txt
```

View [example](Example.ipynb) for the brief info.

The [pytorch_survival script](pytorch_survival.py) contains all numerical experiments, described in the article. To visualize collected data use the [results merging script](merge_script.py) to merge all the experiments results and then draw the figures using the [drawing script](survival_drawing_script.py).
