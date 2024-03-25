...
# 1. Setting up the enviroment for this project

In my first machine learning (ML) project i classified the classic iris flower dataset.
The main reason that i chose this project is because it is well understood even by non ML experts/programmers. 
This problem is a classification problem, allowing to understand and practice the use of easier supervised learning algorithms. 

...
# 2. So first things first; Installing Python libraries
     - Scipy
     - Numpy
     - matplotlib
     - pandas
     - sklearn

     this took me quite a while since i previously hadn't installed these large libraries and pretty much just went with "vanilla" python. However this helped me to train my command line skills and version control. As the project progressed further some libraries were not "working properly", since the installment was not the right version etc. After some Googling and stackoverflow troubleshooting i managed to move forward. 

...
# 3.Load the data

As said in the beginning i used the iris flower dataset on my main.py file. 
The dataset contains 150 observations of iris flower strains. There are four columns of the flowers in centimeters. The fifth column is the species of the flower observed. All observed flowers belong to one of the three species.

...
# 4.Import libraries.
...
# 4.1 Load dataset

This included the URL, names and dataset.

....
# 5 Summarizing the dataset

Here i learned to look at the data a few different ways.
 -Dimensions of the dataset
 -Peek at the data itself
 -Statistical summary of all attributes
 -Breakdown of the data by the class variable


 On the first one we see 150 instances and 5 attributes in the terminal.

 Peek the Data: We see 20 rows of the data.
     sepal-length  sepal-width  petal-length  petal-width        class
0            5.1          3.5           1.4          0.2  Iris-setosa
1            4.9          3.0           1.4          0.2  Iris-setosa
2            4.7          3.2           1.3          0.2  Iris-setosa
3            4.6          3.1           1.5          0.2  Iris-setosa
4            5.0          3.6           1.4          0.2  Iris-setosa
5            5.4          3.9           1.7          0.4  Iris-setosa
6            4.6          3.4           1.4          0.3  Iris-setosa
7            5.0          3.4           1.5          0.2  Iris-setosa
8            4.4          2.9           1.4          0.2  Iris-setosa
9            4.9          3.1           1.5          0.1  Iris-setosa
10           5.4          3.7           1.5          0.2  Iris-setosa
11           4.8          3.4           1.6          0.2  Iris-setosa
12           4.8          3.0           1.4          0.1  Iris-setosa
13           4.3          3.0           1.1          0.1  Iris-setosa
14           5.8          4.0           1.2          0.2  Iris-setosa
15           5.7          4.4           1.5          0.4  Iris-setosa
16           5.4          3.9           1.3          0.4  Iris-setosa
17           5.1          3.5           1.4          0.3  Iris-setosa
18           5.7          3.8           1.7          0.3  Iris-setosa
19           5.1          3.8           1.5          0.3  Iris-setosa


Then There is the Statistical summary. This one includes the count, mean, the min and max values as well as some percetiles. All the numerical values have the same scale (cm) and similar ranges between 0 and 8 cm. 
        sepal-length  sepal-width  petal-length  petal-width
count    150.000000   150.000000    150.000000   150.000000
mean       5.843333     3.054000      3.758667     1.198667
std        0.828066     0.433594      1.764420     0.763161
min        4.300000     2.000000      1.000000     0.100000
25%        5.100000     2.800000      1.600000     0.300000
50%        5.800000     3.000000      4.350000     1.300000
75%        6.400000     3.300000      5.100000     1.800000
max        7.900000     4.400000      6.900000     2.500000

The Class distribution can be viewed as an absolute count. Each class has the same number of instances (50 to 33% of the dataset).
class
Iris-setosa        50
Iris-versicolor    50
Iris-virginica     50

....
# 6. Data Visualization

This was the most exciting part. In the main.py file i started with the univariate plot/plots for each individual variable. 
![Reference Image](/my_venv/images/Figure_1.png)

I tried the histograms of each input variable to get and idea of the distribution.
In the picture it seems like two of the input variables have a Gaussian distribution.
![Refernce Image](/my_venv/images/Figure_2.png)

Scatter plot matrix is a helpful approach to spot structured relationhips between input variables.
![Reference Image](/my_venv/images/Figure_3.png)


...
# 7 Evaluating Algorithms 

 This took me the most of time in the project. Since this was the first project that i used some real algorithms i put on some serious effort to understand the meaning of these different algorithms and how they work. All the coding work from now on was created in algorithms.py file. 

 First i splitted the dataset into two, 80% of which was used to train, evaluate and select among the models and 20% that i hold as a validation dataset. 

 I learned the basics in how to index, slice and reshape numpy arrays for ML.

...
 # 7.1 Building the models

I tried 6 different algorithms:
 -Logistic Regression (LR)
 -Linear discriminant Analysis (LDA)
 -K-Nearest Neighbors (KNN)
 -Classification and Regression trees (CART)
 -Gaussian Naive Bayes (NB)
 -Support Vector Machines (SVM)

 Linear algorithms in this case were LR & LDA and rest were nonlinear alogithms.

 ...
 # 7.2 Selecting best model

 By running the code in algorithms.py file i got the best result for Support Vector Machines (SVM) in the terminal. SVM had the largest estimated accuracy score for this particular model. The test can be run multiple times but the results differ only slightly each time. 

 I compared each algorithms by "create a box and whisker plot" for each distribution and comparing the distributions as follows: 
 ...
 PY comment :Compare Algorithms
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()

This is the most clear figure to me to compare the algorithms and the "breaktrough" moment in the project. The box and whisker plots are squashed at the top of the range, with some evaluations achieving 100% accuracy, and some pushing down into the high 80% accuracies.

![Reference Image](/my_venv/images/Figure_4.png)

...
# 8. Predictions

 As sited on the previous section the SVM was the most accurate model for this case. 

 Lastly i evaluated the predictions and got classification report:
 0.9666666666666667
[[11  0  0]
 [ 0 12  1]
 [ 0  0  6]]
                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        11
Iris-versicolor       1.00      0.92      0.96        13
 Iris-virginica       0.86      1.00      0.92         6

       accuracy                           0.97        30
      macro avg       0.95      0.97      0.96        30
   weighted avg       0.97      0.97      0.97        30

...
# 9. Summary

This project taught me quite a few things about ML.
Machine learning can be viewed as part of three main elements:
 -Data
 -Models
 -Algorithms

I learned briefly about all the different python libraries suited for ML.

