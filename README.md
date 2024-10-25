**Lab 5 Question:**

**Topic:** Implementing CNN on the Fashion-MNIST Dataset

**Objective:** In this lab, you will implement a Convolutional Neural Network (CNN) using
the Intel Image Classification dataset. Your task is to train a CNN model to classify
these images with high accuracy. (Link:-
https://www.kaggle.com/datasets/puneet6060/intel-image-classification)


**Tasks:**

1. Dataset Overview:

- Visualize a few samples from the dataset, displaying their corresponding
labels.
2. Model Architecture:
- Design a CNN model with at least 3 convolutional layers, followed by
pooling layers and fully connected (dense) layers.
- Experiment with different kernel sizes, activation functions (such as
ReLU), and pooling strategies (max-pooling or average pooling).
- Implement batch normalization and dropout techniques to improve the
generalization of your model.

3. Model Training:
- Split the dataset into training and test sets.

- Compile the model using an appropriate loss function (categorical cross-
entropy) and an optimizer (such as Adam or SGD).

- Train the model for a sufficient number of epochs, monitoring the training
and validation accuracy.

4. Evaluation:
- Evaluate the trained model on the test set and report the accuracy.
- Plot the training and validation accuracy/loss curves to visualize the
model's performance.
- Display the confusion matrix for the test set to analyze misclassified
samples.
5. Optimization :
- Experiment with data augmentation techniques (rotation, flipping,
zooming) to further improve the model’s performance.
- Fine-tune hyperparameters like learning rate, batch size, and the number
of filters in each layer.
