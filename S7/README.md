# Creating a Deep Neural Network for the given Task/Target:
## S7_Datasets:
This file has two different function:
1. load_data_S7: In this function we just applied basic transformation to train data i.e resize,totensor and normalization.
whereas the transformation to the test data is totensor and normalization.
This function has been applied to model 4 and model 5.

2. load_data_S7_1: This function is applied to model 1,2 and 3 with transformation to the train data i.e. randomapply,randomrotation,resize,totensor and normalization.
whereas the transformation to the test data is totensor and normalization.

## Target:
Target is same for the all DNN Models
1. 99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
2. Less than or equal to 15 Epochs
3. Less than 8000 Parameters

# Model 1:
### Result:
1. Parameters: 15.7k
2. Best Training Accuracy: 98.62
3. Best Test Accuracy: 99.22

### Analysis:
![image](https://github.com/Omkar1634/ERA_V2_Omkar/assets/64948764/9da4e88f-1161-4e5c-a168-45a8ae41a1b2)


# Model 2:
### Result:
1. Parameters: 10.7k
2. Best Training Accuracy: 96.69
3. Best Test Accuracy: 97.26

### Analysis:
![image](https://github.com/Omkar1634/ERA_V2_Omkar/assets/64948764/77d2b4c3-086f-4d4a-aeb0-b8ab572bebe8)


# Model 3:
### Result:
1. Parameters: 8.4k
2. Best Training Accuracy: 97.55
3. Best Test Accuracy: 98.92

### Analysis:
![image](https://github.com/Omkar1634/ERA_V2_Omkar/assets/64948764/ff2f1673-b98b-42fd-9c92-cf6ca66b1404)


# Model 4:
### Result:
1. Parameters: 5.7k
2. Best Training Accuracy: 95.87
3. Best Test Accuracy: 95.86

### Analysis:
![image](https://github.com/Omkar1634/ERA_V2_Omkar/assets/64948764/e1a9e3ab-80c0-46fe-9c17-e916f60692cc)


# Model 5:
### Result:
1. Parameters: 3.9k
2. Best Training Accuracy: 99.20
3. Best Test Accuracy: 98.75

### Analysis:
![image](https://github.com/Omkar1634/ERA_V2_Omkar/assets/64948764/de2359cb-4d61-42ac-8175-70f1c069b329)
