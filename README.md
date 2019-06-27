##### Prediction module.

###### To run without the docker container:

1) Download the file _datapoints_all.pkl_ (available in Dittlab, distributed as necessary), and put it into the _sampledata_ folder.

2) Make sure MySQL server is running on your host. If necessary, change the appropriate lines 
in the configuration in the _putSampleDataIntoMySQL.py_ file:

```
db = mysql.connect(
    host="localhost",
    user="...",
    passwd="...",
    database="mytrac"
)
```

3) Run _putSampleDataIntoMySQL.py_. The sample data is now in your MySQL db in the necessary format.

4) Run _startServer.py_.

5) Your delay prediction server is now running. You can verify it by navigating your browser to 

http://localhost:5000/

You should see the following message:

```
Delay Prediction Server is running!
```

###### Usage documentation

When the server starts, it automatically checks the database and finds all unique values of
_linedir_ - an identification of a particular PT route (line number and direction combined).
Then it reads the datapoints for each linedir, creates, trains and saves to disk the trained
ML model.

If, for some reason, you want to rerun the training for a particular linedir, you can use the
following command:

```
http://localhost:5000/training?parameter1=value1&parameter2=value2&...
```

The following parameters (all of them optional) can be submitted to the training request:

**linedir** : Unique _linedir_ for which to perform the re-training, e.g. "13-1"
 
**from_date** : Starting date from which to use data for training, in the format "yyyymmdd" e.g. "00000000" (for all days), "20150301" 

**to_date** : End date before which to use data for training, in the format "yyyymmdd" e.g. "30001231" (for all days), "20150324" 



The main usage of the module is for prediction requests. Each such request can provide you with
a prediction of the duration and the type (cluster) of the existing bunching swings formation.

The request can be sent in the following form:

```
http://localhost:5000/prediction?parameter1=value1&parameter2=value2&...
```

The following parameters are accepted:

**linedir** : Unique _linedir_ for which to perform the re-training, e.g. "13-1"
 
Date and time of the moment at which to perform the prediction. If absent, the current date and
time will be used (normal mode of operations), but earlier dates or times can be used for 
testing or other purposes. 
 
**date** in the format "yyyymmdd" e.g. "20150327" 

**time** in the format of the number of seconds since the beginning of the day, e.g. 
46971 for 13h 2m 51s

The prediction's response will be in the following format, where **cluster** represents the type
of cluster (as described in the Deliverable 3.2), and the **duration** represents the predicted
duration in seconds:

```
{"cluster": "3", "duration": "3818"}
```

If no bunching is observed, both values will be set to -1.

Examples of prediction requests: 

```
http://localhost:5000/prediction?linedir=1-0&date=20150303&time=36800


http://localhost:5000/prediction?linedir=1-0&date=20150303&time=46800


http://localhost:5000/prediction?linedir=1-0&date=20150301&time=46800
```




