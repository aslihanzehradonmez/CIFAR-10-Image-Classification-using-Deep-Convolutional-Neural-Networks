# CIFAR-10 Image Classification with Deep Convolutional Neural Networks 🖼️

This project demonstrates a complete workflow for image classification using the CIFAR-10 dataset. It includes two main components: a Jupyter Notebook for building and training a Deep Convolutional Neural Network (CNN), and a separate inference notebook to use the trained model for classifying new, user-provided images.

## ✨ Key Features

  * 🧠 **Deep CNN Model:** Implements a robust Convolutional Neural Network architecture for image classification.
  * 🖼️ **CIFAR-10 Dataset:** Utilizes the popular 10-class object recognition dataset.
  * 🛠️ **Data Augmentation:** Employs `ImageDataGenerator` to create augmented image data, improving model generalization.
  * 💪 **Advanced Training:** Integrates Dropout for regularization, Early Stopping to prevent overfitting, and `ReduceLROnPlateau` to dynamically adjust the learning rate.
  * 📊 **Performance Evaluation:** Includes visualizations for training/validation accuracy and evaluates the final model on the test set.
  * 🚀 **Inference Script:** Provides a standalone notebook to load the trained model and classify any custom image.

## 🛠️ Tech Stack

  * 🐍 Python 3
  * 🧠 TensorFlow & Keras
  * 🔢 NumPy
  * 📊 Matplotlib
  * 🖼️ Pillow (PIL)

## ▶️ Quick Start

1.  **Clone the Repository:**

    You can clone the repository to your local machine by running the following command in your terminal:

    ```bash
    git clone https://github.com/aslihanzehradonmez/CIFAR-10-Image-Classification-using-Deep-Convolutional-Neural-Networks.git
    ```

    After cloning, navigate into the project directory:

    ```bash
    cd CIFAR-10-Image-Classification-using-Deep-Convolutional-Neural-Networks
    ```

2.  **Install dependencies:**

    ```bash
    pip install tensorflow numpy matplotlib pillow jupyterlab
    ```

3.  **Run the Jupyter Notebooks:**

    Launch Jupyter and run the notebooks in the following order:

    1.  **Train the Model:** Open and run all cells in `CIFAR-10 Image Classification using Deep Convolutional Neural Networks.ipynb`. This will train the model and save the weights as `cifar10_classifier_model.keras` in the same directory.
    2.  **Classify an Image:** Open and run `cifar10_image_classifier.ipynb`. This notebook will load the saved model and use it to classify a sample image (e.g., `images.jpg`). You can change the `IMAGE_TO_CLASSIFY` variable to test your own images.

## 🏆 Outcome

This project successfully builds, trains, and deploys a Convolutional Neural Network capable of classifying images from the CIFAR-10 dataset with a test accuracy of approximately **81.10%**. The primary outcome is a saved Keras model (`cifar10_classifier_model.keras`) that can be used for inference, as demonstrated by the accompanying classifier notebook. The project effectively showcases the power of CNNs for image recognition and highlights best practices in training, including data augmentation, dropout, and learning rate scheduling to achieve robust performance.