
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from plotnine import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import sys
import itertools
import pickle
#%matplotlib inline

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')
	print(cm)
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

def plotConfusionMatrix(test_labels,predicted_labels):
	cnf_matrix = confusion_matrix(test_labels, predicted_labels)
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=seq2, normalize=True,title='Normalized confusion matrix')
	plt.show()


class sfo(object):
	def __init__(self, trainFile, testFile):
		self.trainFile = trainFile
		self.testFile = testFile
		self.__xgbtrain = None
		self.train_data = None
		self.train_labels = None
		self.category_unique=None
		self.test_data = None
		self.iden = None
		self.predicted_labels = None


	
	def merge(self):
		df = pd.read_csv(self.trainFile,parse_dates=['Dates'])
		df1 = pd.read_csv(self.testFile, parse_dates=['Dates'])
		self.iden=df1['Id']
		frames=[df,df1]
		df2=pd.concat(frames).reset_index(drop=True)
		self.preprocessing(df2)
		df2,self.category_unique = self.extract_features(df2)
		self.train_data = df2[:len(df)].drop('CategoryNum', 1)
		self.train_labels= df2[:len(df)].CategoryNum
		self.test_data = df2[len(df):].drop('CategoryNum', 1)

	def preprocessing(self,data):
		data.loc[data.X>-122.355,'X']=np.median(data.X)
	    	data.loc[data.X<-122.51,'X']=np.median(data.X)
	    	data.loc[data.Y>40,'Y']=np.median(data.Y)

	#visualization for columns using Dates,Address,PdDistrict,DayOfWeek
	#def visualizton_ggplot(self):
	#	g=ggplot(data,aes(x='Category', y='column_name',fill='Category'))+\
    	#	geom_bar(stat='identity',show_legend=False)+coord_flip()+facet_grid("~column_name")
    	#	g.save('n1.png',height=20,width=50,limitsize=False)
	#visualization for features using X and Y columns
	#def visualization_maps(self,data):
	#	sf = mpimg.imread("map.png")
    	#	data.plot(kind="scatter",x='column_name_1',y='column_name_2',alpha=0.01,figsize=(20,20))
    	#	plt.imshow(sf,extent=[min_value_of_column_1,max_value_of_column_1,min_value_of_column_2,max_value_of_column_2])
    	#	plt.savefig('q.png')
    
	
	def extract_features(self,data):
	   	data = data.fillna(0)

	    # Dates
	    	data['Hour'] = data.Dates.dt.hour
	    	data['DayOfWeekNum'] = pd.Series(data.DayOfWeek).astype('category').cat.codes
	    	data['DayOfMonth'] = data.Dates.dt.day
	    	data['Month'] = data.Dates.dt.month
	    	data['Year'] = data.Dates.dt.year
	    	data["Fri"] = np.where(data.DayOfWeek == "Friday",1,0)
	    	data["Sat"] = np.where(data.DayOfWeek == "Saturday",1,0)
	    	data["Weekend"] = data["Fri"] + data["Sat"]
	    	data['StreetName']=data['Address'].str.split().str[-1]
	    	data['street_corner'] = data['Address'].apply(lambda x: 1 if '/' in x else 0)
	    	data['Sc']=1
	    	x=data.groupby(['Dates','Address'])[['Sc']].transform('count')
	    	data.Sc=x.Sc
	    	data['Seasons']=data["Month"].apply(lambda x:self.seasons(x))
	    	data['day_parts']=data["Hour"].apply(lambda y:self.dayparts(y))  
	    # PdDisrict
	    	data['PdDistrictCat'] =pd.Series(data.PdDistrict).astype('category').cat.codes
	  

	    
	    	data["rot_45_X"] = .707*data["Y"] + .707*data["X"]
	    	data["rot_45_Y"] = .707* data["Y"] - .707* data["X"]

	    	data["rot_30_X"] = (1.732/2)*data["X"] + (1./2)*data["Y"]
	    	data["rot_30_Y"] = (1.732/2)* data["Y"] - (1./2)* data["X"]

	    	data["rot_60_X"] = (1./2)*data["X"] + (1.732/2)*data["Y"]
	    	data["rot_60_Y"] = (1./2)* data["Y"] - (1.732/2)* data["X"]

	    	data["radial_r"] = np.sqrt( np.power(data["Y"],2) + np.power(data["X"],2) )
		
	    #clusters
	    	km = KMeans(n_clusters=40)
	    	km.fit(data[['X','Y']])
	    	data['cluster'] = km.labels_

	    # Output feature - crime category
	    
	    	data['CategoryNum'] = pd.Series(data.Category).astype('category').cat.codes


	    	classes = pd.Series(data.Category).astype('category').cat.categories

	    	return pd.concat([data.Hour,
		              pd.get_dummies(data.StreetName,prefix='StreetName'),
		              pd.get_dummies(data.street_corner,prefix='street_corner'),
		              pd.get_dummies(data.Sc,prefix='Sc'),
		              pd.get_dummies(data.Weekend,prefix='weekend'),
		              pd.get_dummies(data.Seasons,prefix='Seasons'),
		              pd.get_dummies(data.day_parts,prefix='dayparts'),
		              data.DayOfWeekNum,
		              data.DayOfMonth,
		              pd.get_dummies(data.Month),
		              pd.get_dummies(data.Year),
		              data.PdDistrictCat,
		              data.rot_45_X,
		              data.rot_45_Y,
		              data.rot_30_X,
		              data.rot_30_Y,
		              data.rot_60_X,
		              data.rot_60_Y,
		              pd.get_dummies(data.cluster,prefix='cluster'),
		              data.radial_r,
		              data.CategoryNum
		                 ], axis=1), classes


	def seasons(self,x):
		if x in range(1,3):
			return "Winter"
	    	if x in range(3,6):
			return "Spring"
	    	if x in range(6,9):
			return "Summer"
	    	if x in range(9,12):
			return "Autumn"
	    	if (x==12):
			return 'winter' 



	def dayparts(self,y):
		if y in range(0,6):
			return "dawn"
	    	if y in range(6,12):
			return "morning"
	    	if y in range(12,18):
			return "evening"
	    	if y in range(18,24):
			return "night"

	def set_param(self):
	    
	    # setup parameters for xgboost
	    	param = {}
	    	param['objective'] = 'multi:softprob'
	    	param['eta'] = 0.4
	    	param['silent'] = 1
	    	param['nthread'] = 4
	    	param['num_class'] = 37
	    	param['eval_metric'] = 'mlogloss'

	    # Model complexity
	    	param['max_depth'] = 8 #set to 8
	    	param['min_child_weight'] = 1
	    	param['gamma'] = 0 
	    	param['reg_alfa'] = 0.05

	    	param['subsample'] = 0.9
	    	param['colsample_bytree'] = 0.8 #set to 1

	    # Imbalanced data
	    	param['max_delta_step'] = 1
	    #use all resources
	    	param['n_jobs']=-1
	    
	    	return param




	def train_xgb(self):
		dtrain = xgb.DMatrix(self.train_data, label=self.train_labels)
		

		param = self.set_param()
		num_round =70
		# Train XGBoost 
		self.__xgb = xgb.train(param, dtrain, num_round);

	def test_xgb(self):
		dtest = xgb.DMatrix(self.test_data)
		# Predict using XGBoost 
		self.predicted_labels=self.__xgb.predict(dtest)




	def csv(self):
		s=pd.DataFrame(self.predicted_labels,columns=self.category_unique,index=self.iden)
		s=s.drop([0],axis=1)
		s.to_csv('sub1_xgb.csv')
		s.head()

	def pickle(self):
		with open ('submission.pkl','wb') as f:
    			pickle.dump(self.__xgb,f)
	

if __name__ == "__main__":
	train_data_name = sys.argv[1]
	test_data_name = sys.argv[2]
	print(test_data_name)
	model = sfo(train_data_name,test_data_name)
	model.merge()
	model.train_xgb()
	model.test_xgb()
	model.csv()
	model.pickle()
	
