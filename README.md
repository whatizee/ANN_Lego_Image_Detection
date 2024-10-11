

# Fully Connected Neural Network Experiments

## Project Overview

This project involves experimenting with various fully-connected neural network architectures on a provided image dataset using Python and Keras in a Google Colab environment. The primary goal is to achieve the highest possible accuracy on the dataset through different network configurations and hyperparameter tuning. This repository includes code, documentation, and the results of multiple experiments.

## Group Members

- Jibin George (0832593)
- Jibin Kuruppassery Sebastian (0829897)
- Kailas Krishnan Radhakrishnan Sudhadevi (0850313)
- Vishal Ramesh Babu (0832438)

## Lab Outline

The lab is divided into two parts:
- **Part 1**: Create the neural network architecture.
- **Part 2**: Conduct experiments and report results.

## What’s Included

This repository contains the following:
- `lab_experiments.ipynb`: The Google Colab notebook with the code for building and training the neural networks, as well as the experiment results.
- `lab_experiments.pdf`: A PDF export of the final notebook with all code and outputs.
- `README.md`: Documentation and instructions for the project.
- `results/`: Folder containing plots for training and validation loss/accuracy from different experiments.

## Dataset

The dataset used for this project consists of grayscale images of various LEGO pieces, classified into six categories:
- Brick 1x2
- Brick 2x2
- Brick 2x4
- Plate 1x2
- Plate 2x2
- Plate 2x4

All images were resized to 128x128 pixels and converted to NumPy arrays. The data was split into training, validation, and testing sets with a 70-20-10 split, respectively.

## Experiments

We conducted multiple experiments to optimize the network performance by adjusting the following aspects:
- Number of layers and neurons.
- Different activation functions.
- Optimizers (e.g., Adam).
- Learning rate adjustments.
- Use of dropout layers to prevent overfitting.
- Batch size variations.
- Initializers for weights and biases.

### Key Experiments

1. **Baseline Model**: A simple network with three hidden layers and ReLU activations.
2. **Experiment with Dropout**: Added dropout layers to reduce overfitting.
3. **Increase in Layers**: Added more layers and increased the number of neurons.
4. **Optimizer Tuning**: Changed the learning rate and optimizer settings.
5. **Weight Initialization**: Experimented with different weight and bias initializers.
6. **Batch Size Adjustments**: Increased the batch size for training.
7. **Final Model**: Selected the best-performing model based on validation accuracy and tested it on the test set.

Each experiment is documented in the notebook with plots showing training and validation loss/accuracy per epoch.

## How to Run the Project

1. **Google Colab**:
   - Open the `lab_experiments.ipynb` file in Google Colab.
   - Ensure that you have access to the dataset in your Google Drive.
   - Run the code cells sequentially to train and evaluate the models.

2. **Requirements**:
   - Python 3.x
   - Keras and TensorFlow
   - NumPy
   - Matplotlib
   - Google Colab

## Results

The best-performing model achieved an accuracy of **X%** on the validation set and **Y%** on the test set. See the `lab_experiments.ipynb` file for detailed results and visualizations.

### Example Plots

![](results/train_val_accuracy_plot.png)
*Training and Validation Accuracy Over Epochs*

![](results/train_val_loss_plot.png)
*Training and Validation Loss Over Epochs*

## Conclusion

Through this project, we gained hands-on experience in designing, tuning, and evaluating fully connected neural networks using Keras. The experiments highlighted the impact of network architecture and hyperparameters on model performance. The final model demonstrated good generalization ability on unseen test data.

## Future Work

- Explore convolutional neural networks (CNNs) for improved performance on image data.
- Implement further hyperparameter optimization techniques like random search or Bayesian optimization.
- Experiment with transfer learning using pre-trained models.

## References

- Keras Documentation: https://keras.io/
- TensorFlow Documentation: https://www.tensorflow.org/
- Google Colab Documentation: https://colab.research.google.com/
