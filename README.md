# csmath

This repository is used to accomplish my homework in math class.There are six files in all, five for five homeworks and the rest one for the final project.
- [01_curve_fitting.py](https://github.com/Jieeee/csmath/blob/master/01_curve_fitting.py)
- [02_pca.py](https://github.com/Jieeee/csmath/blob/master/02_pca.py)
- [03_mog_em.py](https://github.com/Jieeee/csmath/blob/master/03_mog_em.py)
- [04_lm.py](https://github.com/Jieeee/csmath/blob/master/04_lm.py)
- [05_svm.py](https://github.com/Jieeee/csmath/blob/master/05_svm.py)
- [final_project.py](https://github.com/Jieeee/csmath/blob/master/final_project.py)

## 01. Curve fitting
Goal:

Implement polynomial curve fitting in python.

Requirement:

- Programming lanuage: python
- Plot the results in matplotlib

Experiment result:

- red points stand for 10 training data points generated from function f = sin(x)
- blue curve stands for the true function f = sin(x)
- red curve stands for the predicted curve

![image](https://github.com/Jieeee/csmath/blob/master/result/01_figure.png)

## 02. PCA
Goal:

Represent digits '3' in 2D
- convert data from the UCI Optical Recognition of Handwritten Digits Data Set
- perform PCA over all digit '3' with 2 components
- plot the PCA results as below (also in page #12 of PCA)

Requirements:

- Programming lanuage: python
- Plot the results in matplotlib

Experiment result:

![image](https://github.com/Jieeee/csmath/blob/master/result/pca_points.png)
![image](https://github.com/Jieeee/csmath/blob/master/result/pca_result.png)
## 03. MOG and EM
Goal:

implement MOG in 2D case

- Generate 2D Gaussian distribution
- E-M method

Requirements

- Programming lanuage: python
- Plot the results in matplotlib

Experiment result:

![image](https://github.com/Jieeee/csmath/blob/master/result/em_points.png)
## 04. LM algorithm
Goal:

- Implement the Levenberg-Marquardt method

Requirements:

- Programming lanuage: python

## 05. SVM
Goal:

Implement (simplified) SVM method

- input 2D data and their label (in two classes)
- implement quadratic programming
- output (and plot) classification results

Requirements:

- Programming lanuage: python
- Plot the results in matplotlib


##06. Final project
Paper reading and implement:
 
ICCV2015 [Fast and Effective L0 Gradient Minimization by Region Fusion](http://120.52.73.13/www.cv-foundation.org/openaccess/content_iccv_2015/papers/Nguyen_Fast_and_Effective_ICCV_2015_paper.pdf) by RMH Nguyen

Main idea are detailed in the pdf file [final_project_report.pdf](https://github.com/Jieeee/csmath/blob/master/final_project_report.pdf)
