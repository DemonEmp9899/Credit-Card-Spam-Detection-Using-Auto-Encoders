# Credit-Card-Spam-Detection-Using-Auto-Encoders
Credit Card Spam Detection using Denoising Autoencoders and Deep Neural Networks
Objective:
To develop a robust credit card spam detection system by first denoising and extracting meaningful features from noisy transaction data using Denoising Autoencoders (DAE), and then classifying transactions using a Deep Neural Network (DNN) for final spam detection.

Background:
Credit card fraud and spam transactions are serious concerns in the digital banking and financial industry. Traditional models struggle to generalize well due to:

Noisy or incomplete data

High dimensionality of features

Class imbalance (few spam vs. many legitimate transactions)

To address these challenges, we propose a two-stage approach:

Feature Denoising & Compression using Autoencoders

Classification using Deep Neural Networks

Methodology:
1. Data Preprocessing:
Dataset: Publicly available credit card transaction datasets (e.g., Kaggle or UCI)

Steps:

Handling missing values

Normalization (MinMax or Standard Scaling)

Train-test split (typically 80:20)

Addressing class imbalance using SMOTE or class weights

2. Denoising Autoencoder (DAE):
Autoencoders are neural networks that learn compressed representations of data.

Denoising Autoencoder is trained to reconstruct clean data from a corrupted version.

Architecture:

Input layer (same as feature size)

Encoder (e.g., Dense → ReLU)

Bottleneck (low-dimensional compressed feature)

Decoder (mirrored architecture)

Noise Addition: Gaussian noise or dropout added to input features

Loss Function: Mean Squared Error (MSE)

Optimizer: Adam

The output from the encoder (compressed features) is passed to the DNN classifier.

3. Deep Neural Network (DNN) Classifier:
Input: Features learned from the encoder of the DAE

Architecture:

Dense layers with ReLU activation

Dropout layers for regularization

Final layer with sigmoid activation (binary classification: spam or not)

Loss Function: Binary Crossentropy

Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC

Results:
Denoising Effectiveness: Autoencoder effectively reduced noise and dimensionality while retaining important features.

Classification Accuracy: DNN achieved improved performance compared to using raw data directly.

AUC-ROC Score: Demonstrated the model’s capability to distinguish between spam and legitimate transactions even in imbalanced conditions.

Advantages:
Noise-resilient features using DAE lead to more robust classification.

Layered architecture allows for modular training and tuning.

Improved generalization compared to standalone classifiers like logistic regression or decision trees.

Tools and Technologies Used:
Python with libraries: TensorFlow/Keras, NumPy, Pandas, Scikit-learn

Matplotlib/Seaborn for visualization

Google Colab / Jupyter Notebook for implementation and testing

Conclusion:
This hybrid model combining Denoising Autoencoders and Deep Neural Networks significantly improves the ability to detect credit card spam transactions in noisy and imbalanced datasets. This approach can be further enhanced using ensemble methods, real-time streaming data, or hybrid unsupervised-supervised learning frameworks.
