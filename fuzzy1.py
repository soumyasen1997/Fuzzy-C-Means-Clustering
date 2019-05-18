import pandas as pd
import numpy as np
import random
import operator
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

def Membership_initialize():

	#Membership Matrix
    u = []

    for i in range(row):

    	randomlist=[]
   
    	for j in range(c):
    		randomlist.append(random.random())

    	l=[]
    	for j in randomlist:
    		l.append(j/sum(randomlist))

    		#initialize with random numbers between 0 and 1
    	u.append(l)	  
    
    return u



def cluster_center(u):

    cluster_mem_val = list(zip(*u))   #unzip the membership values to calculate the cluster centers
    
    vj=[]

    uij=[]

    for i in range(c):
        
        for j in list(cluster_mem_val[i]):
        	uij.append(j**m)

        #denominator value of cluster center formula
        denominator=sum(uij)

        upper = []

        for i in range(row):
            xpoint = list(data.iloc[i])
            l=[]
            for xi in xpoint:
            	l.append(uij[i] * xi)
            	
            upper.append(l)
        
        #numerator value of cluster center formula
        numerator = map(sum, zip(*upper))

        center=[]

        for k in numerator:
        	center.append(k/denominator) 

        vj.append(center)

    return vj


def Membership_update(u, cluster_center):

    for i in range(row):

        x = list(data.iloc[i])

        #calculate distaces between points i.e dij and dik
        d=[]

        for k in range(c):
        	d.append(np.linalg.norm(list(map(operator.sub, x, cluster_center[k]))))
        #print(d)

        for j in range(c):

        	#calculate the denominator of memership value
        	denominator = sum([(d[j]/d[k])**(2/(m-1)) for k in range(c)])

        	#update the membership value
        	u[i][j]= float(1/denominator)   
    
    return u


def get_cluster_labels(val):

    cluster_labels = []

    for i in range(row):

        max_val, idx = max((val, idx) for (idx, val) in enumerate(val[i]))
        cluster_labels.append(idx)

    return cluster_labels


def fuzzy_CMeans():

    # initialize the membership Matrix
    val = Membership_initialize()
    
    curr = 0
    
    while (max_iteration>0):
    	
    	max_iteration-=1

        cluster_centers = cluster_center(val)

        val = Membership_update(val, cluster_centers)
        
        cluster_labels = get_cluster_labels(val)

   # print(membership_mat)
    return cluster_labels



#driver function

names=[0,1,2,3,4]
data = pd.read_csv("iris.csv",names=names)
data=data.iloc[:,:4]

#number of rows and columns in the dataset 
row,column=data.shape

# Maximum number of iterations
max_iteration = 50

# Fuzzy parameter
m = 3

#number of clusters
c = 3

#function call
labels = fuzzy_CMeans()


pca = PCA(n_components=3).fit(data)
pca_c = pca.transform(data)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors=['red','green','yellow']

for i,a in enumerate(pca_c):
    ax.scatter(a[0],a[1],a[2],c=colors[labels[i]],s=40)

plt.title("Fuzzy C-means clustering")
plt.show()


