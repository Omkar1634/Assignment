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

Training Loss: This graph shows a rapid decrease in training loss, leveling off as epochs increase. The steep decline at the beginning indicates that the model quickly learned from the training data. The leveling off suggests that the model reached a point where additional learning from the training data doesn't significantly reduce the loss anymore. This is a typical desirable pattern, showing that the model's ability to predict the training data improved quickly.

Test Loss: The test loss graph also starts high and drops quickly, but it shows some fluctuation (ups and downs) as epochs increase. This fluctuation might be a sign of the model's instability in generalizing to new, unseen data (test data). However, the overall trend is downward, suggesting improvement in the model's predictive ability on the test set.

Training Accuracy: This graph indicates a sharp increase in training accuracy, which then plateaus. It appears that the model achieved high accuracy quite fast and maintained it throughout the remaining epochs, suggesting the model fits the training data well.

Test Accuracy: The test accuracy graph reveals a more gradual increase compared to the training accuracy. After some fluctuations, it also reaches a plateau, but at a lower level than the training accuracy. This plateau is typical and expected, but the gap between training and test accuracy may point to a slight overfitting, where the model performs better on the training data than on the test data.

Overall, the model seems to be performing well, with both losses decreasing and accuracies increasing. However, the fluctuations in test loss and the lower test accuracy compared to the training accuracy suggest that there might be room for improvement.

# Model 2:
### Result:
1. Parameters: 10.7k
2. Best Training Accuracy: 96.69
3. Best Test Accuracy: 97.26

### Analysis:
![image](https://github.com/Omkar1634/ERA_V2_Omkar/assets/64948764/77d2b4c3-086f-4d4a-aeb0-b8ab572bebe8)
Training Loss: This graph shows a decrease in training loss from the start, indicating that the model is learning and improving its predictions on the training data. The curve flattens out as the epochs increase, suggesting that the model is approaching a minimum loss value where further improvements on the training set are minimal.

Test Loss: Unlike the smooth curve in the training loss, the test loss graph shows significant fluctuations as epochs increase. This could indicate that the model's performance on the test set, which is data it hasn't seen during training, is somewhat unstable. The fluctuations suggest variance in the model's predictions, which could be a sign of overfitting, where the model is too closely fit to the training data and does not generalize well to new data.

Training Accuracy: The training accuracy graph increases sharply initially and then levels off to a high accuracy that is maintained through the subsequent epochs. This indicates that the model has a high degree of accuracy on the training dataset, which is typical as the model is directly learning from this data.

Test Accuracy: The test accuracy graph shows much more variability compared to the training accuracy. While the overall trend is upward, indicating improving accuracy on the test data, the significant ups and downs could again be a sign of overfitting or possibly an issue with how the test data is being presented to the model during each epoch (e.g., if the test data is shuffled or if there's a non-representative selection of data in each epoch).

The graphs together suggest that while the model is learning and improving its performance on training data, its performance on test data is less stable. 


# Model 3:
### Result:
1. Parameters: 8.4k
2. Best Training Accuracy: 97.55
3. Best Test Accuracy: 98.92

### Analysis:
![image](https://github.com/Omkar1634/ERA_V2_Omkar/assets/64948764/ff2f1673-b98b-42fd-9c92-cf6ca66b1404)
Training Loss: The training loss graph shows a significant drop very early in the training process, leveling off quickly to a low loss value. This suggests the model learns most of its necessary features from the data early and doesn't improve much after that. This can be expected if the model is simple or the task is not very complex.

Test Loss: The test loss graph exhibits a generally decreasing trend but includes some variability with slight increases at certain points. It's not as smooth as the training loss graph, which may indicate that the model's predictions are not as stable on unseen data (the test set) as they are on the training data.

Training Accuracy: The training accuracy graph shows a sharp increase and then plateaus at a high level. This suggests that the model is fitting the training data well and that there might be a ceiling effect where the model can't improve on the training set beyond a certain point, possibly because it has effectively memorized the training data.

Test Accuracy: The test accuracy graph is more volatile compared to the training accuracy. Although it trends upward, indicating learning and improvement, the variability suggests that the model's generalization to new data is not consistent across epochs. This could be due to various factors, such as a small test set, inherent noise in the test data, or the model encountering data points it has not effectively learned to handle.

Together, these graphs indicate that while the model is quite effective on the training data, its performance on the test data is not as consistent, which could be a sign of overfitting. Overfitting occurs when a model is too complex relative to the amount and variety of data available and learns the training data too well, including the noise, at the expense of losing generalization power. 


# Model 4:
### Result:
1. Parameters: 5.7k
2. Best Training Accuracy: 95.87
3. Best Test Accuracy: 95.86

### Analysis:
![image](https://github.com/Omkar1634/ERA_V2_Omkar/assets/64948764/e1a9e3ab-80c0-46fe-9c17-e916f60692cc)
Training Loss: This graph shows a typical descent from a high loss at the start of training to a much lower value, leveling off as the epochs increase. The rapid decrease at the beginning suggests the model quickly learned from the training data, and the leveling out indicates a diminishing return on learning from additional epochs.

Test Loss: The test loss graph similarly shows a steep decline, which is positive as it suggests that the model is improving its predictions on the test dataset as well. The curve smoothens out, indicating that the model reaches a stable performance on the test set. The absence of significant fluctuations suggests good generalization at this point.

Training Accuracy: The graph for training accuracy exhibits a sharp increase initially and then plateaus near 95%. This pattern is common and suggests the model is becoming proficient at making correct predictions on the training data.

Test Accuracy: The test accuracy graph displays a steady rise and then levels off, mirroring the pattern of the test loss graph. This indicates that the model's performance on unseen data is improving and stabilizing. It's a good sign that the accuracy plateaus rather than fluctuates, which can be indicative of the model's ability to generalize from the training data to new, unseen data.

These graphs together suggest that the model trains effectively and reaches a stable level of high performance. There isn't a pronounced sign of overfitting, as both the training and test metrics stabilize without a widening gap. The model seems to generalize well, though always room for improvement exists, possibly through fine-tuning or further experimentation with model parameters.




# Model 5:
### Result:
1. Parameters: 3.9k
2. Best Training Accuracy: 99.20
3. Best Test Accuracy: 98.75

### Analysis:
![image](https://github.com/Omkar1634/ERA_V2_Omkar/assets/64948764/de2359cb-4d61-42ac-8175-70f1c069b329)
Training Loss: The training loss graph starts at approximately 0.3 and sharply decreases to below 0.05 within the initial epochs, showing a quick reduction in error on the training dataset. It then levels off, indicating the model's good fit to the training data.

Test Loss: This graph shows the test loss starting just above 0.0012 and then dropping quickly, followed by a couple of small peaks and eventually flattening out. The pattern suggests that the model's performance on the test data improves quickly and then stabilizes, with slight variations possibly due to the model's reaction to new patterns or noise in the test dataset.

Training Accuracy: The training accuracy increases rapidly to above 95% within the first few epochs and then continues to climb more slowly, plateauing close to 98%. This high level of training accuracy indicates that the model is very effective at learning from the training dataset.

Test Accuracy: The test accuracy graph exhibits more variability. It starts at around 97.4%, dips to below 97%, then increases to approximately 98.6% before leveling off and showing some fluctuation. This behavior might suggest some overfitting to the training data, but overall, the test accuracy remains high, which is a positive sign.

In summary, the model shows excellent learning capabilities with high accuracy and low loss on both the training and test datasets. The leveling off in both loss graphs indicates that additional training epochs beyond this point might not yield significant improvements. The slight fluctuations in test loss and accuracy suggest that while the model is generalizing well, there might be some overfitting or instability that could be addressed with techniques such as dropout, regularization, or more varied training data.
