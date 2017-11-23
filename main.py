# -*- coding:utf-8 -*-
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA


def generate_data ( ) :
	num = 100
	mean = [ 0 , 0 ]
	cov = [ [ 10 , 8 ] , [ 2 , 1 ] ]
	x , y = np.random.multivariate_normal ( mean , cov , num ).T
	
	# x = [ 1 , 2 ]
	# y = [ 3 , 4 ]
	return x , y


def plot ( x , y , fig , title ) :
	fig = plt.figure ( fig )
	ax1 = fig.add_subplot ( 111 )
	ax1.set_title ( title )
	plt.ylim ( [ -5 , 5 ] )  # adjust the max leaving min unchanged
	plt.xlabel ( 'X' )
	plt.ylabel ( 'y' )
	ax1.scatter ( x , y , c = 'b' , marker = 'o' , label = title )
	plt.show ( )


def rotate ( x , y ) :
	x_mean = np.mean ( x )
	y_mean = np.mean ( y )
	# 其实这里的mean都是0，不用remove的，不过规范的流程是需要remove
	x_meanRemoved = x - x_mean
	y_meanRemoved = y - y_mean
	
	# 组合两列数据，计算协方差
	combined = np.vstack ( (x_meanRemoved , y_meanRemoved) )
	cov = np.cov ( combined )
	
	eigVals , eigVects = np.linalg.eig ( np.mat ( cov ) )  # 计算特征值和特征向量
	
	eigValInd = np.argsort ( eigVals )  # 对特征值进行排序，默认从小到大
	
	eigValInd = eigValInd [ :-(9999 + 1) :-1 ]  # 逆序取得特征值最大的元素
	
	redEigVects = eigVects [ : , eigValInd ]  # 用特征向量构成矩阵
	
	rotateDataMat = combined.T * redEigVects  # 用归一化后的各个数据与特征矩阵相乘，映射到新的空间
	
	lowDimensionMat = np.asarray ( rotateDataMat [ : , 0 ] )
	
	plot ( combined [ 0 ] , combined [ 1 ] , 1 , "Original Data" )
	
	plot ( np.asarray ( rotateDataMat [ : , 0 ] ) , np.asarray ( rotateDataMat [ : , 1 ] ) , 2 , "Rotated Data" )
	
	plot ( lowDimensionMat , np.array ( [ 0 for i in range ( len ( lowDimensionMat ) ) ] ) , 3 ,
	       "Single Dimension Data" )
	
	f = plt.figure ( 4 )
	ax1 = f.add_subplot ( 311 )
	ax1.set_title ( "Original Data" )
	plt.ylim ( [ -10 , 10 ] )  # adjust the max leaving min unchanged
	plt.xlabel ( 'X' )
	plt.ylabel ( 'y' )
	ax1.scatter ( combined [ 0 ] , combined [ 1 ] , c = 'b' , marker = 'o' , label = "Original Data" )
	
	ax2 = f.add_subplot ( 312 )
	ax2.set_title ( "Rotated Data" )
	plt.ylim ( [ -10 , 10 ] )  # adjust the max leaving min unchanged
	plt.xlabel ( 'X' )
	plt.ylabel ( 'y' )
	ax2.scatter ( np.asarray ( rotateDataMat [ : , 0 ] ) , np.asarray ( rotateDataMat [ : , 1 ] ) , c = 'r' ,
	              marker = 'o' , label = "Rotated Data" )
	
	ax3 = f.add_subplot ( 313 )
	ax3.set_title ( "Single Dimension Data" )
	plt.ylim ( [ -10 , 10 ] )  # adjust the max leaving min unchanged
	plt.xlabel ( 'X' )
	plt.ylabel ( 'y' )
	ax3.scatter ( lowDimensionMat , np.array ( [ 0 for i in range ( len ( lowDimensionMat ) ) ] ) , c = 'y' ,
	              marker = 'o' , label = "Single Dimension Data" )
	
	plt.show ( )
	
	print ( rotateDataMat )


def facial_reduce ( n_comp , sequence_ID ) :
	X = [ ]
	
	for i in range ( 1 , 11 ) :
		path = "./orl_faces/s1/"
		path = path + str ( i ) + ".pgm"
		X.append ( list ( Image.open ( path ).getdata ( ) ) )
	
	X = np.array ( X )
	X = X - np.mean(X)
	
	pca = PCA ( n_components = n_comp , svd_solver = 'full' , whiten = True )
	
	X_project = pca.fit_transform ( X )
	new_X = pca.inverse_transform ( X_project )
	
	for i in range ( 1 , 11 ) :
		img = np.asarray ( new_X [ i - 1 , : ] ).reshape ( 112 , 92 )
		plt.imshow ( img , cmap = plt.cm.gray , interpolation = 'nearest' )
	
		path = './output/n={n_comp}/s{seq_id}/{current_i}.png'.format ( n_comp = n_comp , seq_id = sequence_ID ,
		                                                                current_i = i )
		print ("Currently we are processing " + path)
		plt.savefig ( path )


def facial_management ( ) :
	for i in range ( 1 , 2 ) :
		facial_reduce ( 1 , i )

	for i in range ( 1 , 2 ) :
		facial_reduce ( 5 , i )


def clear_file ( n_comp , sequence_ID ) :
	remove_path = './output/n={n_comp}/s{seq_id}'.format ( n_comp = n_comp , seq_id = sequence_ID )
	if os.path.exists ( remove_path ) :
		shutil.rmtree ( remove_path )
	if not os.path.exists ( remove_path ) :
		os.mkdir ( remove_path )


def clear_op ( ) :
	for i in range ( 1 , 41 ) :
		clear_file ( 1 , i )
	
	for i in range ( 1 , 41 ) :
		clear_file ( 5 , i )


# X = np.array ( X )
# f = plt.figure ( 5 )
# ax = f.add_subplot ( 111 )
# ax.imshow ( X , cmap = plt.cm.binary )
# plt.show ( )


if __name__ == '__main__' :
	# x , y = generate_data ( )
	# rotate ( x , y )
	facial_management ( )
