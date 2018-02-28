# Digit-recognizer-from-scratch-using-Nueral-Networks
Data set= mnist data set  
Accuracy upto 96.78%
>>> import dataloader  
>>> training_data, validation_data, test_data = \  
... dataloader.load_data_wrapper()  
>>> import recognizer  
>>> net = recognizer.Network([784, 30, 10], cost=recognizer.CrossEntropyCost)  
>>> net.SGD(training_data, 30, 10, 0.5,  
... lmbda = 5.0,  
... evaluation_data=validation_data,  
... monitor_evaluation_accuracy=True,  
... monitor_evaluation_cost=True,  
... monitor_training_accuracy=True,  
... monitor_training_cost=True)  
