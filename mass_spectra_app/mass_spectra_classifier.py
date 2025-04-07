import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import copy
import torch.nn as nn

class DistanceClassifier(nn.Module):
    def __init__(self):
        super(DistanceClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

@st.cache_resource()
def load_models():
    try:
        cnn = torch.load('cnn_complete.pth', weights_only=False, map_location=torch.device('cpu'))
        lstm = torch.load('lstm_complete.pth', weights_only=False, map_location=torch.device('cpu'))
        fc = torch.load('fc_complete.pth', weights_only=False, map_location=torch.device('cpu'))
        classifier = torch.load('distance_classifier_full.pth', weights_only=False, map_location=torch.device('cpu'))
        cnn.eval()
        lstm.eval()
        fc.eval()
        classifier.eval()
        return cnn, lstm, fc, classifier
    except FileNotFoundError as e:
        st.error(f"Error loading model files: {e}. Make sure the .pth files are in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during model loading: {e}")
        st.stop()

# Function to preprocess CSV
@st.cache_data()
def preprocess_csv(uploaded_file_content):
    try:
        df = pd.read_csv(uploaded_file_content, skiprows=2, names=['Point', 'mz', 'intensity'])
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None, None # Return None for both raw and processed

    df_raw_for_plot = df.copy() # Keep a copy for plotting before modification

    if 'intensity' not in df.columns or 'mz' not in df.columns:
        st.error("CSV file must contain 'mz' and 'intensity' columns (case-sensitive).")
        return df_raw_for_plot, None # Return raw, but None for processed

    max_intensity = np.nanmax(df['intensity'])

    df['intensity'] = df['intensity'] / max_intensity
    df['intensity'] = (3 * df['intensity']) / ((2 * df['intensity']) + 1)

    bins = np.arange(39, 546 + 1, 1)
    num_bins = len(bins) - 1

    df['bin'] = np.digitize(df['mz'], bins) - 1
    df = df[(df['bin'] >= 0) & (df['bin'] < num_bins)]

    binned_intensities = df.groupby('bin')['intensity'].max()
    intensity_vector = binned_intensities.reindex(range(num_bins), fill_value=0).values

    # Return both the raw df (for plotting) and the processed vector
    return df_raw_for_plot, intensity_vector

# Function to compute embeddings
def spectrum_embedding(x, cnn, lstm, fc):
    x = x.unsqueeze(1)  # Add channel dimension
    x = cnn(x)  # Apply CNN
    x = x.transpose(1, 2)  # Prepare for LSTM
    x, _ = lstm(x)  # Apply LSTM
    x = fc(x[:, -1, :])  # Take last LSTM output
    return x

# Load models (they are now stored in global variables)
cnn, lstm, fc, classifier = load_models()

# Streamlit UI
st.title("Mass Spectrometry Classifier")

# Streamlit UI
col1, col2 = st.columns(2)
prediction, probability = None, None
df1_raw_for_plot, df1_processed = None, None
df2_raw_for_plot, df2_processed = None, None

with col1:
    uploaded_file1 = st.file_uploader("Upload Spectra #1", type="csv", key="file1")
    if uploaded_file1:
        # Preprocess immediately after upload
        df1_raw_for_plot, df1_processed = preprocess_csv(uploaded_file1)

        # Display initial plot (without colored background yet)
        # if df1_raw_for_plot is not None:
        #     st.write("### Spectra #1 (Raw)")
        #     st.line_chart(df1_raw_for_plot.set_index("mz")['intensity'])
        # else:
        #     st.warning("Could not read or process Spectra #1.")
        

with col2:
    uploaded_file2 = st.file_uploader("Upload Spectra #2", type="csv", key="file2")
    if uploaded_file2:
        # Preprocess immediately after upload
        df2_raw_for_plot, df2_processed = preprocess_csv(uploaded_file2)

         # Display initial plot (without colored background yet)
        # if df2_raw_for_plot is not None:
        #     st.write("### Spectra #2 (Raw)")
        #     st.line_chart(df2_raw_for_plot.set_index("mz")['intensity'])
        # else:
        #     st.warning("Could not read or process Spectra #2.")

if df1_raw_for_plot is not None and df2_raw_for_plot is not None:
    # Convert to input tensor format
        x1 = torch.tensor(df1_processed, dtype=torch.float32).unsqueeze(0)
        x2 = torch.tensor(df2_processed, dtype=torch.float32).unsqueeze(0)

        # Compute embeddings
        with torch.no_grad():
            embed1 = spectrum_embedding(x1, cnn, lstm, fc)
            embed2 = spectrum_embedding(x2, cnn, lstm, fc)

        
        # Compute pairwise distance
        distance = F.pairwise_distance(embed1, embed2, p=2).item()
        distance_tensor = torch.tensor(distance, dtype=torch.float32).unsqueeze(0)

        # Pass through classifier
        with torch.no_grad():
            probability = classifier(distance_tensor).item()
        
        # Decision threshold
        threshold = 0.8163
        prediction = "Same" if probability > threshold else "Different"

if df1_raw_for_plot is not None or df2_raw_for_plot is not None:
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        # if prediction == "Same":
        #     prediction_metric_class = "metric-container-same"
        # elif prediction == "Different":
        #     prediction_metric_class = "metric-container-different"
        # else:
        #     prediction_metric_class = None
        # st.markdown(f'<div class="{prediction_metric_class}">', unsafe_allow_html=True)
        st.metric(label="Prediction", value=prediction)
        # st.markdown('</div>', unsafe_allow_html=True)
    with res_col2:
        if probability is not None:
            st.metric(label="Similarity Score", value=f"{probability:.4f}")
        else:
            st.metric(label="Similarity Score", value="—")

    if df1_raw_for_plot is not None and df2_raw_for_plot is None:
        combined_df = pd.DataFrame({
            "mz": df1_raw_for_plot["mz"],
            "First Input": df1_raw_for_plot["intensity"]
        }).set_index("mz")
        
    elif df1_raw_for_plot is None and df2_raw_for_plot is not None:
        combined_df = pd.DataFrame({
            "mz": df2_raw_for_plot["mz"],
            "Second Input": df2_raw_for_plot["intensity"]
        }).set_index("mz")
    else:
        combined_df = pd.DataFrame({
            "mz": df1_raw_for_plot["mz"],
            "Spectra #1": df1_raw_for_plot["intensity"],
            "Spectra #2": df2_raw_for_plot["intensity"]
        }).set_index("mz")

    st.line_chart(combined_df)

# if df1_raw_for_plot is not None and df2_raw_for_plot is not None:
    
#     # Convert to input tensor format
#     x1 = torch.tensor(df1_processed, dtype=torch.float32).unsqueeze(0)
#     x2 = torch.tensor(df2_processed, dtype=torch.float32).unsqueeze(0)

#     # Compute embeddings
#     with torch.no_grad():
#         embed1 = spectrum_embedding(x1, cnn, lstm, fc)
#         embed2 = spectrum_embedding(x2, cnn, lstm, fc)

    
#     # Compute pairwise distance
#     distance = F.pairwise_distance(embed1, embed2, p=2).item()
#     distance_tensor = torch.tensor(distance, dtype=torch.float32).unsqueeze(0)

#     # Pass through classifier
#     with torch.no_grad():
#         probability = classifier(distance_tensor).item()
    
#     # Decision threshold
#     threshold = 0.8163
#     prediction = "Same" if probability > threshold else "Different"
    
#     res_col1, res_col2 = st.columns(2)
#     with res_col1:
#         st.metric(label="Prediction", value=prediction)
#     with res_col2:
#         st.metric(label="Similarity Score", value=f"{probability:.4f}")

#     # st.write("### Combined Plot of Both Spectra")
#     combined_df = pd.DataFrame({
#         "mz": df1_raw_for_plot["mz"],
#         "First Input": df1_raw_for_plot["intensity"],
#         "Second Input": df2_raw_for_plot["intensity"]
#     }).set_index("mz")
#     st.line_chart(combined_df)