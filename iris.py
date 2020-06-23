import pandas as pd
from sklearn import svm

iris = pd.read_csv('IRIS.csv')
iris_shuffled = iris.sample(frac=1)
print('\nFirst ten row of shuffled data is :')
print(iris_shuffled.iloc[:10])
print('\nSpecies column will be used as labels of flowers, I will predict the species of given flower by using linear svm classifier')
iris_shuffled['species'] = iris_shuffled['species'].astype('category')

print('\nBefore training classifier, let get some information about data')
#using 125 of data as train set and 25 of data as test set
train = iris_shuffled[:125]
test = iris_shuffled[125:]

print('\nMean of training set for different columns are:')
print(train.mean())

print('\nStandart deviation of training set for different columns are:')
print(train.std())

print('\nCovariance matrix of features is following')
print(train.cov())

print('\nCorrelation of features is following:')
print(train.corr())
print('\nAs it can be seen there are high correlation between petal width and petal length')

print('\nIn order to standardize the dataset we need to get zero mean and one variance data so subtracting meand and dividing standart deviation gives standardized dataset')
x = (train.iloc[:,:4]-train.iloc[:,:4].mean())/train.iloc[:,:4].std()
y = (train.iloc[:,4])
print('mean of standardized dataset:')
print(x.mean())
print('std of standardized dataset:')
print(x.std())

mean = train.iloc[:,:4].mean()
std = train.iloc[:,:4].std()

#Setting and fitting svm classifier;
lin_clf = svm.LinearSVC()
lin_clf.fit(x, y)
classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

print('\nAfter fitting linear SVM classifier; we can predict outputs of given inputs')
print('\nFor example for randomly selected data inputs are;')
print(train.iloc[36,:4])
print('\nPutting the input to classifier gives output array;')
output_random = lin_clf.decision_function([(train.iloc[36, :4] - mean) / std])
print(output_random)
print('Since maximum value of the array is in '+ str(output_random.argmax()) + '. index and ' +str(output_random.argmax()) + '. element of the class is '+str(classes[output_random.argmax()]))
print('It means the classifier predict the output of given flower as ' + str(classes[output_random.argmax()]))
print('Since the actual class of given flower is '+ str(train.iloc[36,4]) + '; prediction of classifier given randomly selected example is ' + str(str(train.iloc[36,4])==str(classes[output_random.argmax()])))

train_true = 0
for i in range(len(train)):
    output = lin_clf.decision_function([(train.iloc[i,:4]-mean)/std])
    if classes[output.argmax()] == train.iloc[i,4]:
        train_true = train_true +1

print('\ntraining accuracy is ' + str((train_true/125)*100)+ ' % ')


test_true = 0
i = 0
for i in range(len(test)):
    output = lin_clf.decision_function([(test.iloc[i,:4]-mean)/std])
    if classes[output.argmax()] == test.iloc[i,4]:
        test_true = test_true +1

print('test accuracy is ' + str((test_true/25)*100) + ' %')

