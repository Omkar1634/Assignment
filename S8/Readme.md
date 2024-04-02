# Target 
1. Keep the parameter count less than 50000
2. Max Epochs is 20

I have used two Optimizers SGD & Adam to compare the train and test accuracy.

# 1. Network with Group Normalization
### Result: 
1. Parameter: 47,930
### Analysis :
1. SGD Optimizer:
   **Training** : Train: Loss=1.1508 Batch_id=97 Accuracy=56.66
   **Testing**: Test set: Average loss: 0.0027, Accuracy: 5247/10000 (52.47%)
   ![image](https://github.com/Omkar1634/ERA_V2_Omkar/assets/64948764/30c2a765-c1b2-467f-b787-000c5a85e25b)
   
      A. **Training Loss**: This graph shows a general downward trend, indicating that the model is learning and improving over time. The loss decreases as the number of epochs increases, which is expected. There is a spike around the 25th epoch, which could be due to a variety of factors, such as an outlier in the data, a change in the learning rate, or an issue with the batch of data fed to the model at that point.
      
      B. **Test Loss**: The test loss fluctuates more compared to the training loss. While it generally trends downward, there are several increases, indicating that the model may not be as stable on unseen data. This could suggest that the model is overfitting to the training data or that there's a high variance in the test data.
      
      C. **Training Accuracy**: The training accuracy graph shows an improvement in performance over time, which is consistent with the reduction in training loss. However, there's a significant drop in accuracy around the 10th epoch, similar to the spike seen in the training loss graph. This suggests that whatever caused the spike in training loss also affected the model's accuracy.
      
      D. **Test Accuracy**: The test accuracy trends upwards, but with significant variability. The inconsistency in the test accuracy could indicate that the model's generalization to new data isn't stable. This could be a sign of overfitting, where the model performs well on the training data but fails to maintain performance on the test data.
      
      Overall, these graphs suggest that while the model is learning and improving its performance on the training set, it might be struggling to generalize that learning effectively to unseen data, indicated by the variability in the test loss and accuracy. Measures such as regularization, dropout, or revisiting the model's architecture could potentially improve the model's generalization. Also, implementing techniques like cross-validation could give a better indication of how well the model is likely to perform on independent data.


2. Adam Optimizer:
   **Training** : Train: Loss=1.0936 Batch_id=97 Accuracy=61.32
   **Testing**: Test set: Average loss: 0.0021, Accuracy: 6282/10000 (62.82%)
   ![image](https://github.com/Omkar1634/ERA_V2_Omkar/assets/64948764/a793e8c5-ddf3-4538-83a5-af28489503c0)

      A. **Training Loss**: The training loss graph shows an overall downward trend, suggesting that the model is generally learning and improving. However, there are two pronounced spikes, one just before 20 epochs and another just after 30 epochs. These spikes could be indicative of issues like an unstable learning rate, anomalies in the training data, or a problem with the model's ability to generalize from that batch of training data.
      
      B. **Test Loss**: The test loss shows a downward trend but is much noisier compared to the training loss, with several spikes and drops. The general downward trend is a good sign, indicating the model is learning to generalize somewhat. However, the noise might suggest the model's performance on the test set is volatile and may not be learning consistently.
      
      C. **Training Accuracy**: The training accuracy graph shows a general upward trend, which is what we want to see. It indicates that the model's predictions are becoming more accurate over time on the training data. However, similar to the training loss graph, there are significant drops around the 20th and just after the 30th epochs. These drops reflect the spikes in the training loss graph and suggest the model encountered some issue at these points.
      
      D. **Test Accuracy**:  The test accuracy is increasing, which suggests that the model is becoming better at correctly predicting the unseen data. However, like the test loss, the test accuracy is quite noisy, with significant variability in the accuracy from epoch to epoch. This can be a sign of the model not being robust to variations in the test data.
      
      To summarize, while the model is showing improvement over time on both the training and test data, there are signs of instability, particularly at certain epochs where there are spikes in loss and drops in accuracy. This could be due to various reasons like overfitting, need for better hyperparameter tuning, data issues, or potentially the need for a more sophisticated model architecture. It would be beneficial to explore these areas to improve the model's performance and ensure it is learning stably and consistently.

3. MissClassified Images:
   ![image](https://github.com/Omkar1634/ERA_V2_Omkar/assets/64948764/b11706e0-a8f3-4ef4-832e-8116a899aa03)



# 2. Network with Layer Normalization 
### Result: 
1. Parameter: 45,242
### Analysis :
1. SGD Optimizer:
   **Training** : Train: Loss=1.8531 Batch_id=97 Accuracy=27.61
   **Testing**: Test set: Average loss: 0.0036, Accuracy: 2788/10000 (27.88%)
  ![image](https://github.com/Omkar1634/ERA_V2_Omkar/assets/64948764/8317ea74-b603-4d61-8f7b-7fcfb6cb5d4f)

      A. **Training Loss**: The loss decreases smoothly, which is a positive indication that the model is learning as expected from the training data. There is no sign of erratic changes or spikes, suggesting that the training process is stable.
      
      B. **Test Loss**: The test loss also generally decreases, indicating that the model's performance is improving on the test set as well. The smooth decline without erratic jumps suggests that the model is generalizing well and not overfitting.
      
      C. **Training Accuracy**: The accuracy increases steadily, which is consistent with the declining training loss. The model's predictions on the training data are becoming more accurate over time.
      
      D. **Test Accuracy**: The test accuracy shows an overall upward trend, but it's not as smooth as the training accuracy. There are some fluctuations, which is common in test metrics as they reflect the model's performance on unseen data. The general improvement is a good sign, though the variability suggests there may be room for improving the model's robustness.
      
      The absence of large spikes in loss or dramatic dips in accuracy is encouraging, suggesting that the model is learning effectively without major instability issues. However, it is always important to consider other factors like the complexity of the data, the model's capacity, and whether the model has plateaued in learning or could benefit from further training or hyperparameter tuning.


2. Adam Optimizer:
   **Training** : Train: Loss=1.8330 Batch_id=97 Accuracy=28.96
   **Testing**: Test set: Average loss: 0.0036, Accuracy: 2798/10000 (27.98%)
   ![image](https://github.com/Omkar1634/ERA_V2_Omkar/assets/64948764/814d1641-c7b7-45b5-96a4-e715a1153aae)

      A. **Training Loss**: The training loss graph shows a smooth and steady decline, which is an indicator of good learning progression. The model consistently minimises the error on the training dataset as the epochs increase. No sudden spikes are present, which suggests that there are no significant issues during training.
      
      B. **Test Loss**: The test loss graph, similar to the training loss, indicates a steady decrease in loss over time. This is a positive sign, showing that the model is generalizing well to new data. The smooth decrease without drastic fluctuations suggests that the model is not experiencing high variance when evaluated against the test set.
      
      C. **Training Accuracy**: The training accuracy graph shows a sharp increase initially, which then levels off. It seems that the model quickly learns to predict the training data correctly but then plateaus, which may indicate that it has reached its capacity in terms of what it can learn from the training data.
      
      D. **Test Accuracy**: The test accuracy graph displays an initial sharp increase, followed by fluctuations and an eventual plateau. The variability here is a bit higher than in the training accuracy, which is common since the test set consists of unseen data. However, the plateau suggests the model may not be improving significantly beyond a certain point.
      
      The accuracy metrics, while generally improving, suggest that there may be some overfitting occurring since the training accuracy is higher and less variable than the test accuracy. This might be addressed by techniques like introducing regularization or collecting more diverse training data. It could also be worth examining if the model has reached its capacity and consider experimenting with a more complex model architecture if the problem demands it.
         
3. MissClassified Images: 
   ![image](https://github.com/Omkar1634/ERA_V2_Omkar/assets/64948764/7a6f1963-acde-4060-8c88-7ca4aed2818c)


# 3. Network with Batch Normalization
### Result: 
1. Parameter: 47,650
### Analysis :
1. SGD Optimizer:
   **Training** : Train: Loss=0.8335 Batch_id=97 Accuracy=67.21
   **Testing**: Test set: Average loss: 0.0020, Accuracy: 6407/10000 (64.07%)

   ![image](https://github.com/Omkar1634/ERA_V2_Omkar/assets/64948764/c33aa03b-de50-4c9e-a201-25301427a41a)

A. **Training Loss**: The training loss graph depicts a smooth and consistent decline from the first epoch to the last. This indicates that the model is learning from the training data effectively, with no visible anomalies or erratic spikes.

B. **Test Loss**: The test loss shows a steep decline initially, followed by some fluctuations but still trending downwards overall. This pattern is typical as the model starts to generalize from training data to unseen data. Some variability is expected; however, the overall downward trend is a positive sign.

C. **Training Accuracy**: The graph displays a steady upward trend, indicating increasing accuracy on the training data as the model learns. There’s no sign of erratic behaviour, and the improvement appears consistent.

D. **Test Accuracy**: The test accuracy is improving, showing that the model's predictive performance on unseen data is getting better. However, there is noticeable volatility, with some peaks and troughs, though the overall direction is upward. This volatility could be indicative of the model not being entirely robust to the variability in the test data, which is not uncommon.

The accuracy graphs are improving steadily, although the test accuracy shows some fluctuations. This could suggest the model is performing well but might benefit from techniques to improve its stability and generalization, such as cross-validation, regularization, or adjusting hyperparameters.



2. Adam Optimizer:
   **Training** : Train: Loss=0.7658 Batch_id=97 Accuracy=71.87
   **Testing**: Test set: Average loss: 0.0019, Accuracy: 6808/10000 (68.08%)

   ![image](https://github.com/Omkar1634/ERA_V2_Omkar/assets/64948764/c7566c7e-35d6-499e-b7a0-221a7d3d9884)

A. **Training Loss**: The training loss graph depicts a strong downward trend, leveling off as epochs increase, which is typical as the model begins to converge and there is less room for improvement on the training set.

B. **Test Loss**: The test loss decreases significantly initially and then oscillates, with some upward trends, suggesting periods where the model's performance on the test set has worsened. This could indicate some overfitting or the model's sensitivity to the specific test data.

C. **Training Accuracy**: The training accuracy graph shows a steady increase, leveling off as the epochs increase, which again is typical in the later stages of training as the model starts to saturate in its learning capacity on the training data.

D. **Test Accuracy**: The test accuracy improves markedly up to around the 15th epoch and then exhibits some fluctuations, with a general upward trend. The fluctuations are less pronounced than in the test loss graph, indicating that while the model's predictions on the test set are not perfectly stable, the overall trend is positive.

the training loss and accuracy here show a smoother convergence, and while the test accuracy is more volatile than the training accuracy, it still shows an overall improvement. The fluctuations in test loss and accuracy indicate the model might be reacting to nuances in the test data, which could be addressed by fine-tuning the model, perhaps through hyperparameter optimization or using techniques like dropout or batch normalization to improve generalization. The goal is to have a model that not only fits the training data well but also generalizes effectively to new, unseen data.
3. MissClassified Images:
  ![image](https://github.com/Omkar1634/ERA_V2_Omkar/assets/64948764/ed4b1d26-9585-443c-abef-f2ae9feb90e8)

