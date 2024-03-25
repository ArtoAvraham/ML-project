import pandas as pd
# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)
...
# shape
print(dataset.shape)
...
# head
print(dataset.head(20))
...
# descriptions
print(dataset.describe())

# class distribution 
print(dataset.groupby('class').size())

# visualize the data
# box and whisker plots
from matplotlib import pyplot as plt
dataset.plot(kind='box', subplots=True, layout=(2,2),sharex=False, sharey=False)
plt.show()

# histograms
dataset.hist()
plt.show()

# scatter plain matrix
from pandas.plotting import scatter_matrix
scatter_matrix(dataset)
plt.show()