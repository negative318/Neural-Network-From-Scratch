# Week 3
# I.  Steps to make a neural network model
1.	Identify the problem and collect data
    -	Identify the problem: Need to clearly understand the problem that want to solve.
    -	Collect data: Collect data related to the problem to be solved. The data needs to be large enough and responsive to the problem.
2.	Data preprocessing
    -	Clean data: Eliminate missing values, handle outliers, and remove unnecessary fields.
    -	Standardized data: Normalize values to the same data type that the machine can learn such as vector, matrix, ...
    -	Data division: Divide the data into training, testing, and validation sets.
3.	Build a neural network model
    -	Choose the architecture: Choose the number of layers, number of nodes for each layer and propagation type (Fully Connected, convolutional, recurrent, …).
    -	Choose the activation function: Make the data non-linear such as ReLU,Sigmoid, Tanh, …
    -	Choose the loss function: Choose the appropriate loss functions for the model you need to train such as Cross-Entropy, Binary-Cross-Entropy for classification or Mean Squared Error (MSE), Mean Absolute Error (MAE) for regression.
4.	Model training
    -	Choose the optimal algorithm: Gradient Descent (GD), Stochastic Gradient Descent (SGD), …
    -	Set parameters: Choose learning rate, number of epochs, batch size.
    -	Training model: Use training data to train the model and through validation test to update hyper parameters accordingly
5.	Evaluate the model
    -	After training model, check model performance on the test set to evaluate generalization ability.

# II. So sánh giữa các parameter khác nhau
1.  So sánh hiệu suất khi thay đổi Unit của hidden layer  
        Active: ReLU, learning rate: 0,01, batch_size: 128
    
    | Unit | train | test |
    |:----------:|:--------:|:--------:|
    | 64 | 96.88% | 90.68% |
    | 128 | 96.09% | 91.58% |
    | 256 | 94.53% | 93.18% |
    | 512 | 99.22% | 93.34% |

2.  So sánh hiệu suất khi thay đổi số hidden layer  
        Active: ReLU, learning rate: 0.01, batch_size: 128

    | num_hidden | Unit | train | test |
    |:----------:|:--------:|:--------:|:--------:|
    | 0 | 0 | 91.41% | 87.66% |
    | 1 | 64 | 96.88% | 90.68% |
    | 2 | 128,64 | 89.01% | 88.02% |
    | 3 | 256,128,64 | 74.21% | 66.78% |

3.  So sánh hiệu suất khi thay đổi các hàm active  
        learning rate: 0.01, batch_size: 128, unit: 128
    
    | Active | train | test |
    |:----------:|:--------:|:--------:|
    | relu | 94.53% | 91.71% |
    | sigmoid | 95.31% | 91.25% |
    | tanh | 88.28% | 86.12% |

4.  So sánh hiệu suất khi thay đổi learning_rate  
        Active: ReLU, batch_size: 128, unit: 128
    
    | learning_rate | train | test |
    |:----------:|:--------:|:--------:|
    | 0.001 | 92.97% | 87.14% |
    | 0.01 | 92.18% | 91.75% |
    | 0.1 | 99.22% | 94.96% |
    | 1 | 1 | 95.73% |

5.  So sánh hiệu suất khi thay đổi batch_size  
    Active: ReLU, learning rate: 0.01, unit: 128

    | batch_size | train | test |
    |:----------:|:--------:|:--------:|
    | 64 | 98.44% | 93.11% |
    | 128 | 92.97% | 91.73% |
    | 256 | 94.92% | 90.71% |
    | 512 | 92.97% | 89.89% |

6.  So sánh hiệu suất khi thay đổi epochs  
    Active:ReLU, learning rate: 0.01, unit: 128, batch_size: 128

    | epochs | train | test |
    |:----------:|:--------:|:--------:|
    | 1000 | 92.97% | 91.94% |
    | 2000 | 94.53% | 93.06% |
    | 3000 | 94.53% | 93.68% |
    | 4000 | 95.60% | 93.71% |