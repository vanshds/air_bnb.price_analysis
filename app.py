import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.markdown(
    """
    <style>
    .stApp {
          background: linear-gradient(to bottom right, #000000, #434343);
        background-attachment: fixed;
        margin: 0 !important;
        padding: 0 !important;
        font-weight: bold !important;
    }

    .block-container {
        padding-top: 0rem !important;
    }
    header, footer {visibility: hidden;}

    label, .stSelectbox label, .stNumberInput label {
        color: white !important;
        font-weight: bold !important;
    }

    /* 🔥 White Title with Drop Shadow */
    h1, h2 {
        color: white !important;
        font-weight: bold !important;
        text-shadow: 2px 2px 4px #00000066;
    }

    .stSuccess {
        font-weight: bold;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ✅ White title with drop shadow
st.title("🏡 Airbnb Price Predictor (NYC)")
@st.cache_data
def load_data():
    df = pd.read_csv('AB_NYC_2019.csv')
    df = df[df['price'] < 1000]
    df = df.dropna(subset=['reviews_per_month'])

    features = ['room_type', 'neighbourhood_group', 'minimum_nights',
                'number_of_reviews', 'reviews_per_month',
                'availability_365']
    df = df[features + ['price']]
    return df, features

@st.cache_resource
def train_model(df, features):
    X = pd.get_dummies(df[features], drop_first=True)
    y = np.log1p(df['price'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    return model, scaler, X.columns.tolist()

df, features = load_data()
model, scaler, feature_columns = train_model(df, features)

room_type = st.selectbox("Room Type", df['room_type'].unique())
neigh_group = st.selectbox("Neighbourhood Group", df['neighbourhood_group'].unique())
minimum_nights = st.number_input("Minimum Nights", min_value=1, value=3)
number_of_reviews = st.number_input("Number of Reviews", min_value=0, value=10)
reviews_per_month = st.number_input("Reviews per Month", min_value=0.0, value=1.2)
availability_365 = st.number_input("Availability (days/year)", min_value=0, max_value=365, value=100)

input_data = {
    'minimum_nights': minimum_nights,
    'number_of_reviews': number_of_reviews,
    'reviews_per_month': reviews_per_month,
    'availability_365': availability_365,
    f'room_type_{room_type}': 1,
    f'neighbourhood_group_{neigh_group}': 1
}

input_df = pd.DataFrame([input_data])
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_columns]

input_scaled = scaler.transform(input_df)

if st.button("Predict Price"):
    pred_log = model.predict(input_scaled)[0]
    pred_price = np.expm1(pred_log)
    st.success(f"💰 Predicted Nightly Price: ₹{pred_price:.2f}")

st.markdown("---")
st.header("📄 Project Analysis Report")

with open("Analysis.pdf", "rb") as pdf_file:
    st.download_button(
        label="Download Analysis PDF",
        data=pdf_file,
        file_name="Analysis.pdf",
        mime="application/pdf"
    )
