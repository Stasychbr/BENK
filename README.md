# BENK method
This repository contains the code base for "The Beran Estimator with Neural Kernels for Estimating the Heterogeneous Treatment Effect" article. The program is written in python 3 using the Pytorch framework.

To satisfy the dependencies use pip and requirements.txt:

```python
pip install -r requirements.txt
```

Also, you need to fix the functions.py file in the sksurv library. Just replace the original file with the [one](modification/functions.py) in the modification folder.

View [example](Example.ipynb) for the brief info.

The [pytorch_survival script](pytorch_survival.py) contains all numerical experiments, described in the article. To visualize collected data use the [drawing script](survival_drawing_script.py).