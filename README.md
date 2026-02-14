# Digit Recognizer (MNIST) – Computer Vision Project

## 🔗 Project Resources

**Hugging Face Deployment (Live Application)**  
This is the deployed web application where users can upload handwritten digit images and receive real-time predictions along with probability distributions.  
https://huggingface.co/spaces/leyuzak/Digit-Recognizer-Computer-Vision  

**Kaggle Training Notebook (Model Development & Submission)**  
This notebook contains the full training pipeline, preprocessing steps, model architecture implementation, evaluation, and Kaggle submission generation.  
https://www.kaggle.com/code/leyuzakoksoken/digit-recognizer-mnist-computer-vision  

---

## 📌 Abstract

This project presents a deep learning–based handwritten digit recognition system developed using the MNIST dataset. A Convolutional Neural Network (CNN) architecture was designed, trained, and evaluated to classify 28×28 grayscale images into ten digit categories (0–9).  

The trained model achieves high classification accuracy and is deployed as an interactive web application. Additionally, the project includes a Kaggle competition pipeline for model evaluation and submission.

---

## 📖 Introduction

Handwritten digit recognition is a fundamental problem in computer vision and pattern recognition. The MNIST dataset serves as a benchmark dataset for evaluating classification models due to its simplicity and standardized format.

The objective of this project is to:

- Design and train a robust CNN model
- Optimize performance for high classification accuracy
- Export the trained model for reuse
- Deploy the model in a web-based interactive interface
- Generate competition-ready submissions for Kaggle

---

## 🧠 Methodology

### Dataset
- MNIST handwritten digits dataset
- 28×28 grayscale images
- 10 class labels (0–9)

### Preprocessing
- Normalization of pixel values to range [0,1]
- Reshaping input data to (28, 28, 1)
- One-hot encoding of target labels

### Model Architecture
The Convolutional Neural Network consists of:

- Convolutional layers with ReLU activation
- MaxPooling layers for spatial feature reduction
- Fully connected (Dense) layers
- Softmax output layer (10 classes)

The architecture is optimized to balance computational efficiency and high classification accuracy.

---

## 📊 Results

The trained model achieves strong predictive performance:

- Kaggle Public Score: **0.99596**
- Accurate digit classification across unseen test data
- Confident probability distribution outputs in deployment interface

The results demonstrate the effectiveness of convolutional architectures for image classification tasks.

---

## 🚀 Deployment

The model is deployed using Hugging Face Spaces, allowing users to:

- Upload digit images
- Receive predicted class output
- Visualize class probability distribution (0–9)

This step highlights model portability and real-world usability beyond experimental environments.

---

## 📂 Project Structure

- `app.py` – Web deployment script  
- `digit-recognizer-mnist-computer-vision.ipynb` – Training and evaluation notebook  
- `mnist_cnn.h5` – Saved model (HDF5 format)  
- `mnist_cnn.keras` – Saved model (Keras format)  
- `submission.csv` – Kaggle submission file  
- `submission_score.png` – Kaggle score visualization  

---

## 🧩 Reproducibility

To run the project locally:

```bash
pip install tensorflow numpy pandas pillow matplotlib streamlit gradio
```

Then launch the application:

If using Streamlit:
```bash
streamlit run app.py
```

If using Gradio:
```bash
python app.py
```

---

## 🎯 Conclusion

This project demonstrates a complete deep learning workflow for computer vision:

- Data preprocessing  
- CNN-based model design  
- Performance evaluation  
- Model serialization  
- Deployment to production-like environment  
- Competition benchmarking  

It serves as both an academic example of applied deep learning and a practical portfolio project showcasing model training, evaluation, and deployment capabilities.

---

## 📜 License

This project is developed for academic and portfolio purposes.
