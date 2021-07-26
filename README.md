This project is an attempt to identify large scale botnet attacks on/using IoT devices in the production field.

Methodology(in brief): As it's not a feasible option to develop a solution on large scale IoT devices, we have a .NET project to simulate the IoT environment. The .NET project sends the messages to an established MQTT topic in the AWS(registered as an IoT Thing in IoT Core). The payload contains the sensor ID, time of message generation and description. Here the description can be of temperature, humidity or any value that a sensor captures. 

An IoT rule is then setup to collect those messages from the topic and deliver to IoT Analytics channel. IoT Analytics is a managed service offering from AWS which is used to collect, process, store and analyse IoT data in real-time. We transform the messages in this stage and set up a trigger every 15 minutes to have all the messages grouped in past 15 minutes. Messages are then extracted to a .csv file and this triggers the containerized notebook which holds the iot_notebook.py code. 

First the pre-processing is ran to identify the counts of messages received from each sensor ID/sensor groups from the .csv file(past 15 minutes) and it is then fed to an LSTM network along with their previous counts. As we divide the dataset to train and test with 80/20 respectively, the latest counts always end up in test set. The LSTM, when ran predicts the counts for the sensor groups for the test set. So when an un-usual jump in counts of messages (possibly triggered by an attack) in the current run is observed, it will result in a high test error - which is the key for us to conduct in-depth analysis on the particular IoT device group for the root cause. 

The following image is the architecture of this project.
![image](https://user-images.githubusercontent.com/57434195/127003592-f818f628-0c13-4744-83d0-f6d2e463cc74.png)
