# YVR-Airport-Object-Detection
Data-driven solution built for YVR Hackathon 2024 to enhance airport safety and efficiency using machine learning, computer vision, and data visualization for real-time anomaly detection and predictive insights.


YVR-Hackathon-2024/
│
├── data/  
│   ├── raw/                      # Original datasets (flight data, weather, operations logs)  
│   ├── processed/                # Cleaned and feature-engineered data for modeling  
│
├── notebooks/  
│   ├── EDA.ipynb                 # Exploratory Data Analysis and data cleaning  
│   ├── Modeling.ipynb            # ML model training and evaluation (Regression, Classification, Clustering)  
│   ├── DeepLearning.ipynb        # CNN and LSTM models for vision and time-series tasks  
│
├── models/  
│   ├── trained_models/           # Saved model weights and serialized ML pipelines  
│   ├── model_evaluation/         # Performance metrics and confusion matrices  
│
├── dashboard/  
│   ├── powerbi/                  # Power BI files and visual dashboards  
│   ├── tableau/                  # Tableau dashboards and data sources  
│
├── src/  
│   ├── data_pipeline.py          # Scripts for data ingestion and ETL automation  
│   ├── feature_engineering.py    # Feature extraction and transformation logic  
│   ├── train_model.py            # Model training, validation, and saving  
│   ├── predict.py                # Inference script for running predictions  
│
├── app/  
│   ├── streamlit_app.py          # Web interface for model results and dashboards  
│   ├── api/                      # Flask/FastAPI endpoints for real-time predictions  
│
├── requirements.txt              # Python dependencies  
├── README.md                     # Project documentation  
└── LICENSE                       # License information  
