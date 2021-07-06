#!/usr/bin/env python
# coding: utf-8

# In[ ]:


print("Hello from containerized conda python 3 - iot_notebook ")
#backup code
#data3.iloc[[2,3,4,7],[1]] = 29
#data3
#data2 = [data2, data3]
#data2 = pd.concat(data2)
#data2_bkp = data2.copy()


# In[ ]:


# installation of tensorflow and keras - only to run on the start of this notebook.
get_ipython().system('pip install tensorflow==2.2')
get_ipython().system('pip install keras')


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import boto3


# In[471]:


iota = boto3.client('iotanalytics')
dataset = "iot_dataset"
dataset_url = iota.get_dataset_content(datasetName = dataset,versionId = "$LATEST")['entries'][0]['dataURI']
data = pd.read_csv(dataset_url)


# In[473]:


data2 = data.copy()
data2 = data2.groupby(by = 'id', axis = 0, as_index = False ).count()
data2.drop(['description','ctimestamp', '__dt'], axis=1, inplace = True)
data2.columns = ['id', 'count']
data2[ 'group' ] = data2['id'] // 10
data2['new_group']  =  + data2['group'].astype(str) +'0' +'-' +  data2['group'].astype(str) +'9'
data2.drop(['id', 'group'], axis = 1, inplace = True)
data2 = data2.groupby(by = 'new_group', as_index = False).sum('count')


# In[553]:


#push current run counts to Run1_CT table in dynamoDB
dynamodb = boto3.resource('dynamodb', region_name='ap-southeast-2', endpoint_url="https://dynamodb.ap-southeast-2.amazonaws.com")
table = dynamodb.Table('Run1_CT')
tstmp = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
for index, row in data2.iterrows():
 #print(index,row['new_group'], row['count'])
 
 eg = {"timestamp" : tstmp, "id" : row['new_group'], "count" : row['count']}
 table.put_item(Item=eg) 


# In[555]:


eg


# In[534]:


#combine the inout data along with last 20 runs
input_data = [input_data, data2]
input_data = pd.concat(input_data)
input_data = input_data.iloc[:]
input_data["count"]=input_data["count"].apply(pd.to_numeric)


# In[ ]:


message = ''
for each in range(1,10):
    print('LSTM for '+ str(each) + '0-'+str(each)+'9')
    input_data2 = input_data[input_data['new_group'] == str(each) + '0-'+str(each)+'9']
    input_data2 = input_data2['count']
    dataset = input_data2.values
    dataset = dataset.astype('float32') 
    #plt.plot(dataset)
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = dataset.reshape(-1,1)
    dataset = scaler.fit_transform(dataset)
    train_size = int(len(dataset) * .8)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    #Use TimeseriesGenerator to organize training data into the right format
    #We can use a generator instead......
    from keras.preprocessing.sequence import TimeseriesGenerator # Generates batches for sequence data
    seq_size = length =  2
    batch_size = 1
    train_generator = TimeseriesGenerator(train,train,length=length,batch_size=batch_size)
    print("Total number of samples in the original training data = ", len(train)) # 95
    print("Total number of samples in the generated data = ", len(train_generator)) # 55
    #With length 40 it generated 55 samples, each of length 40 (by using data of length 95)

    # print a couple of samples... 
    x, y = train_generator[0]

    #Also generate validation data
    validation_generator = TimeseriesGenerator(test, test, length=length ,batch_size=batch_size)


    num_features = 1 
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(length, num_features)))
    model.add(LSTM(50, activation='relu'))
    #model.add(Dense(32))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.summary()
    print('Train...')
    model.fit_generator(generator=train_generator, verbose=2, epochs=100, validation_data=validation_generator)
    trainPredict = model.predict(train_generator)
    testPredict = model.predict(validation_generator)

    trainPredict = scaler.inverse_transform(trainPredict)
    trainY_inverse = scaler.inverse_transform(train)
    testPredict = scaler.inverse_transform(testPredict)
    testY_inverse = scaler.inverse_transform(test)
    trainScore = math.sqrt(mean_squared_error(trainY_inverse[length:], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))

    testScore = math.sqrt(mean_squared_error(testY_inverse[length:], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    if(testScore>20):
        message = message + 'test score for '+ str(each) + '0-'+str(each)+'9' + ' is ' + str(round(testScore,2))+'\n'


# In[467]:


trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[length:len(trainPredict)+length, :] = trainPredict
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
#testPredictPlot[len(trainPredict)+(seq_size*2)-1:len(dataset)-1, :] = testPredict
testPredictPlot[len(train)+(length)-1:len(dataset)-1, :] = testPredict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[550]:


message


# In[556]:


#publish msg to sns topic in case of test error>20
sns = boto3.client('sns')
# Publish a simple message to the specified SNS topic
if(message != ''):
    message = message + '\n' + 'Check the dynamoDB table - Run1_CT for timestamp: '+ tstmp
    response = sns.publish(
        TopicArn='arn:aws:sns:ap-southeast-2:776234713800:IoT_publish_to',   
        Message=message,   
        Subject = 'Higher counts received for devices'
    )


# In[ ]:





# In[ ]:




