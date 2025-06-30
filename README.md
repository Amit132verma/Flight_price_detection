
# ✈️ Flight Price Prediction

This project uses machine learning to predict flight ticket prices based on various parameters like airline, source, destination, total stops, departure and arrival times, and more.

## 📁 Project Structure

```
📦 Flight-Price-Prediction
├── Flight_price_prediction.ipynb
├── README.md
└── requirements.txt
```

## 🚀 Features

* Data preprocessing and feature engineering
* Model training and evaluation
* Price prediction using regression models
* Visualization of key insights from the dataset

## 📊 Technologies Used

* Python 🐍
* Pandas & NumPy
* Scikit-learn
* Seaborn & Matplotlib
* Jupyter Notebook

## 📌 Dataset

The dataset contains details about airline flights, including:

* **Date of Journey**
* **Airline**
* **Source/Destination**
* **Route**
* **Total Stops**
* **Duration**
* **Price (Target)**

> Note: The dataset was preprocessed to extract useful features like journey day/month, departure/arrival hour/minute, duration in minutes, etc.

## 🧠 Models Trained

* **Linear Regression**
* **Random Forest Regressor**
* **Gradient Boosting Regressor**

## 📈 Results

Random Forest Regressor yielded the best performance based on R² score and RMSE.

## 🛠 How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/Flight-Price-Prediction.git
   cd Flight-Price-Prediction
   ```

2. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:

   ```bash
   jupyter notebook Flight_price_prediction.ipynb
   ```

## 🧪 Example Usage

You can use the trained model to predict the price of a flight ticket by inputting the relevant features, such as:

```python
model.predict([[Airline_encoded, Source_encoded, Destination_encoded, Total_Stops, Duration, etc.]])
```

## 📌 Future Improvements

* Hyperparameter tuning with GridSearchCV
* Integration with a web interface using Flask or Streamlit
* Real-time flight data ingestion

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

Would you like me to generate a `requirements.txt` file based on your notebook too?
