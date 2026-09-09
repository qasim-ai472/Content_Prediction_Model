````markdown
# 🎬 Netflix Content Type Prediction Model

A Machine Learning classification project that predicts whether a Netflix title is a **Movie** or a **TV Show** based on selected content features provided by the user.

This project was developed as my **second Machine Learning project during my internship at Auspify Technologies**, with a focus on understanding the complete workflow from data preprocessing and model training to deploying the trained model as a web application.

---

## 📌 Project Overview

Netflix contains a large variety of movies and TV shows with different characteristics such as genre, rating, country, release year, and duration.

The goal of this project is to build a **Binary Classification Model** that learns patterns from Netflix content data and predicts the content type:

- **Movie**
- **TV Show**

The trained model is integrated into a **Flask web application**, where users can enter the required features through a simple HTML/CSS interface and receive a prediction.

---

## 🎯 Project Objective

The main objective of this project is to develop and deploy a machine learning classification model capable of predicting the type of Netflix content based on available features.

The project follows a complete machine learning workflow:

1. Data loading
2. Data preprocessing
3. Feature selection
4. Categorical data encoding
5. Model training
6. Model evaluation
7. Model serialization
8. Flask backend integration
9. Web-based prediction

---

## ✨ Features

- 🎬 Predicts **Movie or TV Show**
- 📊 Uses multiple Netflix content features
- 🧹 Data preprocessing and feature handling
- 🔢 Categorical variable encoding
- 🤖 Machine Learning classification using Scikit-learn
- 💾 Saves the trained model using Joblib
- 🌐 Flask-based backend
- 🎨 HTML & CSS frontend
- ⚡ Real-time prediction through a web interface
- 📦 JSON-based data handling

---

## 🧾 Input Features

The application takes the following information from the user:

| Feature | Description |
|---|---|
| Genre | Genre/category of the Netflix title |
| Rating | Content rating |
| Country | Country associated with the title |
| Release Year | Year in which the title was released |
| Duration | Duration of the content |

Based on these features, the model predicts whether the title is a **Movie** or a **TV Show**.

---

## 🧠 Machine Learning Approach

This project is implemented as a **binary classification problem**.

### Target Variable

The target variable represents the type of Netflix content:

```text
Movie
TV Show
````

### Feature Processing

The input data contains both categorical and numerical features. Before training the model, the data is processed to make it suitable for machine learning.

The workflow includes:

* Selecting relevant features
* Handling input data
* Encoding categorical variables
* Preparing the training data
* Training the classification model
* Evaluating the model
* Saving the trained model

---

## 🛠️ Technologies & Libraries

### Programming Language

* **Python**

### Machine Learning

* **Scikit-learn**

### Data Processing

* **Pandas**
* **Regex (`re`)**
* **JSON**

### Backend

* **Flask**

### Model Persistence

* **Joblib**

### Frontend

* **HTML**
* **CSS**

### Other

* **OS**

---

## 📂 Project Structure

The project follows a structure similar to:

```text
Netflix-Content-Type-Prediction/
│
├── app.py
├── model.py
├── model.pkl
├── data/
│   └── netflix_titles.csv
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── requirements.txt
└── README.md
```

> File names and folder structure may vary depending on the final version of the project.

---

## 🔄 Project Workflow

```text
Netflix Dataset
       │
       ▼
Data Preprocessing
       │
       ▼
Feature Selection
       │
       ▼
Categorical Encoding
       │
       ▼
Model Training
       │
       ▼
Model Evaluation
       │
       ▼
Save Trained Model
       │
       ▼
Flask Backend
       │
       ▼
HTML/CSS Frontend
       │
       ▼
User Input
       │
       ▼
Movie / TV Show Prediction
```

---

## 🌐 Flask Web Application

The trained machine learning model is connected to a Flask backend.

Users can access a web interface where they provide:

```text
Genre
Rating
Country
Release Year
Duration
```

The Flask application processes the submitted information, passes it to the trained model, and returns the predicted content type.

### Local Application

The application runs locally on:

```text
http://127.0.0.1:5000/
```

> This is a local development address and is only accessible when the Flask application is running on your machine.

---

## 🚀 Installation & Setup

Follow the steps below to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/qasim-ai472/Netflix-Content-Type-Prediction.git
```

Navigate to the project directory:

```bash
cd Netflix-Content-Type-Prediction
```

---

### 2. Create a Virtual Environment

It is recommended to use a virtual environment.

```bash
python -m venv venv
```

Activate the virtual environment.

#### Windows

```bash
venv\Scripts\activate
```

#### macOS / Linux

```bash
source venv/bin/activate
```

---

### 3. Install Dependencies

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install the main dependencies manually:

```bash
pip install pandas scikit-learn flask joblib
```

---

### 4. Run the Flask Application

Run the Flask application:

```bash
python app.py
```

After starting the application, open your browser and visit:

```text
http://127.0.0.1:5000/
```

---

## 🖥️ Using the Application

Once the Flask application is running:

1. Open the application in your browser.
2. Enter the required Netflix title information.
3. Provide the genre.
4. Select or enter the rating.
5. Enter the country.
6. Enter the release year.
7. Enter the duration.
8. Submit the form.
9. The trained model will process the input.
10. The application will display the predicted content type.

The final prediction will be either:

```text
Movie
```

or

```text
TV Show
```

---

## 📊 Model Prediction

The prediction process can be represented as:

```text
User Input
    │
    ├── Genre
    ├── Rating
    ├── Country
    ├── Release Year
    └── Duration
          │
          ▼
   Data Preprocessing
          │
          ▼
    Feature Encoding
          │
          ▼
   Trained ML Model
          │
          ▼
      Prediction
          │
          ▼
  Movie / TV Show
```

---

## 💾 Model Serialization

After training, the machine learning model is saved using **Joblib**.

This allows the trained model to be loaded later without retraining it every time the Flask application starts.

The general workflow is:

```text
Training
   ↓
Trained Model
   ↓
Joblib
   ↓
Saved Model File
   ↓
Flask Application
   ↓
Prediction
```

---

## 🔍 Key Concepts Practiced

Through this project, I worked with several important Machine Learning concepts:

* Binary Classification
* Feature Selection
* Data Preprocessing
* Categorical Encoding
* Model Training
* Model Evaluation
* Model Serialization
* Backend Integration
* Web-based Machine Learning Prediction

---

## 📚 Learning Outcomes

This project helped me understand how a machine learning model can be transformed from a training experiment into a usable application.

I gained practical experience in:

* Preparing data for machine learning
* Selecting meaningful features
* Working with categorical variables
* Training classification models with Scikit-learn
* Saving and loading trained models
* Connecting ML models with Flask
* Handling user input from a web form
* Building a simple HTML/CSS interface
* Creating an end-to-end ML prediction workflow

---

## 🎓 Internship Project

This project was developed as part of my **Machine Learning Internship at Auspify Technologies**.

It is my **second project** in the internship journey and focuses on implementing a complete classification workflow rather than only training a model.

The project helped me bridge the gap between **Machine Learning model development and practical application deployment**.

---

## 🔮 Future Improvements

Some possible improvements for future versions include:

* Improving model accuracy through hyperparameter tuning
* Comparing multiple classification algorithms
* Adding more Netflix-related features
* Improving the frontend design
* Adding prediction probability/confidence
* Deploying the application online
* Adding better input validation
* Adding model performance visualizations
* Building a more interactive user interface

---

## 📌 Important Note

This project is intended for **educational and learning purposes**. The prediction depends on the features and dataset used during model development and should not be considered an official Netflix classification system.

---

## 👨‍💻 Author

**Qasim Ali**

Machine Learning / AI Engineering Enthusiast

GitHub:
[https://github.com/qasim-ai472](https://github.com/qasim-ai472)

---

## 🏢 Internship

**Auspify Technologies**

This project was developed as part of my Machine Learning internship journey at Auspify Technologies.

---

## ⭐ Support

If you find this project useful or interesting, consider giving the repository a ⭐ on GitHub.

---

## 🏷️ Tags

```text
Machine Learning
Python
Scikit-learn
Flask
Pandas
Classification
Netflix
Data Science
AI
Artificial Intelligence
ML Project
Machine Learning Project
```

```
```

