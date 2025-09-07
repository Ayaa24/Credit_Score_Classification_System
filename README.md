**Credit Score Classification App**
This project demonstrates a machine learning pipeline to predict a customer's credit score based on their financial and demographic information. The project uses a Neural Network (Multi-Layer Perceptron) for the classification task and a Streamlit web application to provide a user-friendly interface for making predictions.

**Dataset**
The model was trained on the Credit Score Classification Dataset from Kaggle. The dataset contains various features, including a customer's income, loan type, and credit history.

**Prerequisites**
To run this project, you need to have Python installed on your system. You will also need to install the project dependencies.

**Setup and Installation**
   1-Clone the repository:

      git clone [https://github.com/](https://github.com/)[your-username]/[your-repository-name].git

  2-Navigate to the project directory:

     cd [your-repository-name]

  3-Install the required libraries:

    pip install -r requirements.txt

**Running the Application**
To start the Streamlit web application, simply run the following command in your terminal from the project's root directory:

     streamlit run app.py

This will launch the app in your default web browser, where you can input new data and get a real-time credit score prediction.

##Model Training
The trained model files are included in this repository for convenience. However, if you wish to retrain the model, you can run the training script:

python credit_score_nn_classifier.py

This script will re-train the Neural Network and save new model files, which can then be used by the Streamlit application.
