#Author: Sanjukta Aich
import matplotlib.pyplot as plt
import pylab
import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import tree

def RF_train(X_train, Y_train, d, n):
	clf=[]
	for i in range(n):
		idx=np.random.choice(len(Y_train),len(Y_train),replace=True)
		x=X_train[idx,:]
		y=Y_train[idx]
		temp=tree.DecisionTreeClassifier(max_depth=d)
		clf.append(temp.fit(x,y))
	return clf

def OOB_error(X_train, Y_train, d):
	clf=[]
	err=[]
	n=np.round(np.linspace(1,300,num=100))
	for k in n:
		res=[]
		for i in range(int(k)):
			idx=np.random.choice(len(Y_train),len(Y_train),replace=True)
			x=X_train[idx,:]
			y=Y_train[idx]
			temp=tree.DecisionTreeClassifier(max_depth=d)
			clf.append(temp.fit(x,y))
			idx=np.unique(idx)
			oob=[]
			for j in range(len(Y_train)):
				if j not in idx:
					oob.append(j)
			r=[0,0,0,0]
			for j in oob:
				temp=clf[i].predict(X_train[j,:])
				r[int(temp[0])]+=1
			idx=r.index(max(r))
			res.append(idx)	
		temp=len([i for i, j in zip(res,Y_train[oob]) if i!=j])
		k=int(k)
		err.append(temp/k)
	plt.figure(5, figsize=(8,6))
	plt.plot(n,err)
	plt.ylabel('OOB Error')
	plt.xlabel('Number of Trees (Max Depth=full)')
	plt.show()

def RF_test(clf,X_test,l, n):
	res=[]
	for j in range(l):
		r=[0,0,0,0]
		for i in range(n):
			temp=clf[i].predict(X_test[j,:])
			r[int(temp[0])]+=1
		idx=r.index(max(r))
		res.append(idx)
	return res

def Train_Test_err(X_train,Y_train,X_test,Y_test):
	n=np.round(np.linspace(1,1000,num=300))
	err1=[]
	err2=[]
	err3=[]
	err4=[]
	for i in n:
		clf=RF_train(X_train, Y_train, len(Y_train), int(i))
		res=RF_test(clf,X_train,len(Y_train), int(i))
		temp=len([i for i, j in zip(res,Y_train) if i!=j])
		err1.append(temp/len(Y_train))
		res=RF_test(clf,X_test,len(Y_test), int(i))
		temp=len([i for i, j in zip(res,Y_test) if i!=j])
		err3.append(temp/len(Y_test))

		clf=RF_train(X_train, Y_train, 5, int(i))
		res=RF_test(clf,X_train,len(Y_train), int(i))
		temp=len([i for i, j in zip(res,Y_train) if i!=j])
		err2.append(temp/len(Y_train))
		res=RF_test(clf,X_test,len(Y_test), int(i))
		temp=len([i for i, j in zip(res,Y_test) if i!=j])
		err4.append(temp/len(Y_test))

	plt.figure(1, figsize=(8,6))
	plt.plot(n,err1)
	plt.ylabel('Train Error')
	plt.xlabel('Number of Trees (Max Depth=full)')
	plt.figure(2, figsize=(8,6))
	plt.plot(n,err2)
	plt.ylabel('Train Error')
	plt.xlabel('Number of Trees (Max Depth=5)')
	plt.figure(3, figsize=(8,6))
	plt.plot(n,err3)
	plt.ylabel('Test Error')
	plt.xlabel('Number of Trees (Max Depth=full)')
	plt.figure(4, figsize=(8,6))
	plt.plot(n,err4)
	plt.ylabel('Test Error')
	plt.xlabel('Number of Trees (Max Depth=5)')
	plt.show()

wine = np.mat(np.loadtxt("WINE/wine.txt", delimiter=","))
X=wine[:,1:]
Y=wine[:,0]
num_trees=100;
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.75)
clf=RF_train(X_train, Y_train, len(Y_train), num_trees)
res=RF_test(clf,X_test,len(Y_test), num_trees)
target_names = ['class 1', 'class 2', 'class 3']
print(classification_report(Y_test, res, target_names=target_names))

Train_Test_err(X_train,Y_train,X_test,Y_test)

cm=confusion_matrix(Y_test, res)
print(cm)
OOB_error(X_train, Y_train, len(Y_train))