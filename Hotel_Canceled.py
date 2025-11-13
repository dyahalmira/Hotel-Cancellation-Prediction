import streamlit as st
import pandas as pd
import numpy as np
import pickle
import dill
import matplotlib.pyplot as plt
import shap # Pastikan shap di-import
from io import BytesIO

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Hotel Booking Cancellation Predictor",
    page_icon="🏨",
    layout="wide"
)

# --- Fungsi Pemuatan Model dan Explainer ---
@st.cache_resource
def load_model_and_explainer():
    """Memuat model dan explainer dari file."""
    try:
        with open('final_model_RandomForest_Capstone3_20251113_1125.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('tree_explainer.dill', 'rb') as f:
            explainer = dill.load(f)
        return model, explainer
    except FileNotFoundError as e:
        # Menangani jika file tidak ditemukan
        raise FileNotFoundError(f"File tidak ditemukan: {e.filename}")
    except Exception as e:
        # Menangani error pemuatan lainnya
        raise Exception(f"Gagal memuat model/explainer: {e}")

try:
    model, explainer = load_model_and_explainer()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False

# --- Judul Halaman ---
st.title("🏨 Hotel Booking Cancellation Predictor")
st.markdown("---")

# --- Konten Aplikasi ---
if model_loaded:
    # Buat kolom untuk bagian input
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📊 Booking Details")
        
        country = st.selectbox(
            "Country",
            options=['PRT', 'GBR', 'USA', 'ESP', 'IRL', 'FRA', 'DEU', 'ITA', 'BEL', 'NLD', 
                    'CHN', 'BRA', 'CHE', 'AUT', 'POL', 'SWE', 'CZE', 'DNK', 'RUS', 'ROU',
                    'NOR', 'FIN', 'ISR', 'TUR', 'AUS', 'SGP', 'IND', 'JPN', 'KOR', 'NZL',
                    'ZAF', 'ARG', 'MEX', 'CHL', 'ARE', 'SAU', 'EGY', 'MAR', 'UKR', 'GRC',
                    'HKG', 'TWN', 'BGD', 'PAK', 'LBN', 'IRN', 'IRQ', 'IDN', 'THA', 'MYS',
                    'VNM', 'PHL', 'Other']
        )
        
        market_segment = st.selectbox(
            "Market Segment",
            options=['Online TA', 'Offline TA/TO', 'Direct', 'Groups', 'Corporate', 
                    'Complementary', 'Aviation', 'Undefined']
        )
        
        deposit_type = st.selectbox(
            "Deposit Type",
            options=['No Deposit', 'Non Refund', 'Refundable']
        )
        
        customer_type = st.selectbox(
            "Customer Type",
            options=['Transient', 'Transient-Party', 'Contract', 'Group']
        )
        
        reserved_room_type = st.selectbox(
            "Reserved Room Type",
            options=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'L', 'P']
        )
    
    with col2:
        st.subheader("📅 Timing & Changes")
        
        days_in_waiting_list = st.number_input(
            "Days in Waiting List",
            min_value=0,
            max_value=400,
            value=0,
            step=1
        )
        
        previous_cancellations = st.number_input(
            "Previous Cancellations",
            min_value=0,
            max_value=26,
            value=0,
            step=1
        )
        
        booking_changes = st.number_input(
            "Booking Changes",
            min_value=0,
            max_value=21,
            value=0,
            step=1
        )
    
    with col3:
        st.subheader("🎯 Requirements")
        
        required_car_parking_spaces = st.number_input(
            "Required Car Parking Spaces",
            min_value=0,
            max_value=8,
            value=0,
            step=1
        )
        
        total_of_special_requests = st.number_input(
            "Total Special Requests",
            min_value=0,
            max_value=5,
            value=0,
            step=1
        )
    
    st.markdown("---")
    
    # Tombol Prediksi
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button("🔮 Predict Cancellation", use_container_width=True)
    
    if predict_button:
        # 1. Buat input dataframe
        input_data = pd.DataFrame({
            'country': [country],
            'market_segment': [market_segment],
            'deposit_type': [deposit_type],
            'customer_type': [customer_type],
            'reserved_room_type': [reserved_room_type],
            'previous_cancellations': [previous_cancellations],
            'booking_changes': [booking_changes],
            'days_in_waiting_list': [days_in_waiting_list],
            'required_car_parking_spaces': [required_car_parking_spaces],
            'total_of_special_requests': [total_of_special_requests]
        })
        
        try:
            # 2. Lakukan Prediksi
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            
            # --- TAMPILAN PERKIRAAN PERSENTASE PEMBATALAN (Perbaikan di sini) ---
            st.markdown("---")
            st.subheader("🎯 Prediction Result")
            
            col_pred_left, col_pred_center, col_pred_right = st.columns([1, 2, 1])
            with col_pred_center:
                # Tampilkan hasil prediksi
                if prediction == 1:
                    st.error(f"⚠️ **High Risk of Cancellation**")
                    st.metric("Cancellation Probability", f"{prediction_proba[1]:.1%}")
                else:
                    st.success(f"✅ **Booking Likely to Proceed**")
                    st.metric("Cancellation Probability", f"{prediction_proba[1]:.1%}")
                
                # Probability bar
                st.progress(float(prediction_proba[1]))

            st.markdown("---")
            
            # --- Penjelasan SHAP (Dipertahankan dari perbaikan sebelumnya) ---
            st.subheader("🔍 Prediction Explanation (SHAP Tree)")
            
            with st.spinner("Generating explanation..."):
                # 3. Transformasi data input
                preprocessed_output = model.named_steps['preprocessing'].transform(input_data)

                # Dapatkan array dan nama fitur
                if isinstance(preprocessed_output, pd.DataFrame):
                    preprocessed_data_array = preprocessed_output.values
                    feature_names = preprocessed_output.columns.tolist()
                elif isinstance(preprocessed_output, np.ndarray):
                    preprocessed_data_array = preprocessed_output
                    try:
                        feature_names = model.named_steps['preprocessing'].get_feature_names_out(input_data.columns).tolist()
                    except AttributeError:
                        st.warning("Feature names tidak ditemukan. Menggunakan indeks fitur.")
                        feature_names = [f'Feature {i}' for i in range(preprocessed_data_array.shape[1])]
                else:
                    raise TypeError("Output Preprocessing bukan DataFrame atau NumPy array.")

                # 4. Generate SHAP explanation
                shap_values = explainer.shap_values(preprocessed_data_array) 

                # Dapatkan nilai SHAP untuk kelas positif (kelas 1)
                if isinstance(shap_values, list):
                    shap_vals = shap_values[1][0] 
                else:
                    shap_vals = shap_values[0, :, 1]
                
                # Dapatkan expected value untuk kelas positif (kelas 1)
                if isinstance(explainer.expected_value, (list, np.ndarray)):
                    expected_value = explainer.expected_value[1]
                else:
                    expected_value = explainer.expected_value
                
                data_for_explanation = preprocessed_data_array[0]

                # 5. Buat waterfall plot
                fig, ax = plt.subplots(figsize=(10, 6))
                
                explanation = shap.Explanation(
                    values=shap_vals,
                    base_values=expected_value,
                    data=data_for_explanation,
                    feature_names=feature_names
                )
                
                # Menggunakan waterfall_plot
                shap.waterfall_plot(explanation, max_display=10, show=False)
                
                # Tampilkan di Streamlit
                st.pyplot(fig)
                plt.close(fig)
                
                # Additional explanation details
                with st.expander("📊 View Feature Contributions"):
                    contrib_df = pd.DataFrame({
                        'Feature': feature_names,
                        'SHAP Value': shap_vals
                    }).sort_values('SHAP Value', key=abs, ascending=False)
                    st.dataframe(contrib_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error during prediction or explanation: {e}")
            st.exception(e)

else:
    st.warning("⚠️ Model and explainer files not loaded. Please ensure the following files are in the same directory:")
    st.code("""
    - final_model_RandomForest_Capstone3_20251113_1125.pkl
    - tree_explainer.dill
    """)

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p style='color: gray;'>Hotel Booking Cancellation Prediction System</p>
    </div>
    """,
    unsafe_allow_html=True
)