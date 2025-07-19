# wheather-forcast
🌦️ Germany Weather Forecast App

An interactive weather forecast dashboard for 5 major German cities — built with Streamlit, powered by a custom machine learning model trained on historical data from Visual Crossing.

🔮 Features
🏙️ City selector: Berlin, Hamburg, Munich, Frankfurt, Cologne
📅 7-day forecast using ML prediction
🌡️ Current temperature, humidity, wind, and precipitation
🎨 Weather-dependent backgrounds (sunny, rainy, cloudy)
📈 Clean, responsive dashboard-style layout (like Apple Weather)
📸 Preview
<img src="https://user-images.githubusercontent.com/your-screenshot.png" width="600"/>
🚀 Live Demo
🌐 View on Streamlit Cloud

(Replace with actual URL after deployment)
🧠 Tech Stack
Python
Streamlit
scikit-learn
XGBoost
pandas / joblib
Visual Crossing Weather Data
📂 Project Structure
├── app.py                     # Streamlit app
├── train_model.py            # ML model training script
├── weather_data.csv          # Historical weather data
├── weather_forecast_model.pkl
├── model_features.pkl        # Feature order for prediction
├── requirements.txt
└── README.md
⚙️ How to Run Locally
Clone the repo:
git clone git@github.com:your-username/your-repo-name.git
cd your-repo-name
Create a virtual environment (optional but recommended):
python3 -m venv .venv
source .venv/bin/activate
Install dependencies:
pip install -r requirements.txt
Run the app:
streamlit run app.py
📈 Train the Model (Optional)
If you want to retrain the model:

python train_model.py
👥 Team & Credits
Developed by: sahar
Data Source: Visual Crossing
📄 License
This project is licensed under the MIT License.


