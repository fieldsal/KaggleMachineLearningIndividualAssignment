#import dependencies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from category_encoders.target_encoder import TargetEncoder
from sklearn.ensemble import RandomForestRegressor

#training data and test data
#changed file names as they were very long
train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')

target=train_data['Income in EUR']
train_data_id=train_data['Instance']
test_data_id=test_data['Instance']
train_data.drop(['Income in EUR','Instance'],axis=1,inplace=True)


#train_catagorics
train_data["Gender"]=train_data["Gender"].fillna("N/A")
train_data["University Degree"]=train_data["University Degree"].fillna("N/A")
train_data["Hair Color"]=train_data["Hair Color"].fillna("N/A")
train_data["Profession"]=train_data["Profession"].fillna("N/A")
#train_numerics
train_data["Age"]=train_data["Age"].fillna(train_data['Age'].mean())
train_data["Year of Record"]=train_data["Year of Record"].fillna(train_data['Year of Record'].mean())

#test_catagorics
test_data["Gender"]=test_data["Gender"].fillna("N/A")
test_data["University Degree"]=test_data["University Degree"].fillna("N/A")
test_data["Hair Color"]=test_data["Hair Color"].fillna("N/A")
test_data["Profession"]=test_data["Profession"].fillna("N/A")
#test_numerics
test_data["Age"]=test_data["Age"].fillna(test_data['Age'].mean())
test_data["Year of Record"]=test_data["Year of Record"].fillna(test_data['Year of Record'].mean())

numerics_and_catagorics=list(train_data.columns)
X_train_data,X_val,y_train_data,y_val=train_test_split(train_data,target,test_size=0.2,random_state=97)
random_forest=RandomForestRegressor()
#target_encoder
target_encoder=TargetEncoder()
train_data_encoded=target_encoder.fit_transform(X_train_data[numerics_and_catagorics], y_train_data)
test_data_encoded=target_encoder.transform(X_val[numerics_and_catagorics],y_val)
test_data=target_encoder.transform(test_data[numerics_and_catagorics])

print(train_data_encoded.head())
print(test_data_encoded.head())

#random_forest_regressor
r_train_data=random_forest.fit(train_data_encoded,y_train_data)
prediction=r_train_data.predict(test_data_encoded)

def rmse(prediction,target):
	#difference = prediction-target
	#square = **2
	#mean = .mean()
	#root = np.sqrt
	return np.sqrt(((prediction-target)**2).mean())

print(rmse(y_val,prediction)) 
prediction=r_train_data.predict(test_data)

output_template='submission_template.csv'
data_frame=pd.read_csv(output_template)
data_frame['Income']=prediction
data_frame.to_csv('submission.csv',index=False)






