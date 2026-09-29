import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import plotly.express as px
import io

# Set page config
st.set_page_config(page_title="Federated Placement Predictor", layout="wide", page_icon="🎓")


# Super Model Architecture (5 Input Features -> Placement Probability)
class PlacementSuperModel(nn.Module):
    def __init__(self):
        super(PlacementSuperModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


@st.cache_resource
def get_model_instance():
    return PlacementSuperModel()


# UI Header
st.title("🎓 Federated College Placement Analytics & Prediction System")
st.markdown("### Super-Model Synthesis Engine across Decentralized College Networks")
st.divider()

# Sidebar for metadata management
st.sidebar.header("📁 Metadata Upload Center")
st.sidebar.info("Upload `.pt` metadata extracted from College A and College B.")

uploaded_files = st.sidebar.file_uploader(
    "Drag and drop college metadata files",
    type=["pt"],
    accept_multiple_files=True
)

super_model = get_model_instance()
is_model_ready = False

if uploaded_files and len(uploaded_files) >= 2:
    st.sidebar.success(f"✓ Received {len(uploaded_files)} College Metadata files")

    # Process metadata files
    client_dicts = []
    college_names = []

    for file in uploaded_files:
        bytes_data = io.BytesIO(file.read())
        data_payload = torch.load(bytes_data, map_location=torch.device('cpu'))
        client_dicts.append(data_payload["weights"])
        college_names.append(data_payload.get("college_id", file.name))

    # Federated Averaging (FedAvg) aggregation
    super_dict = {}
    first_dict = client_dicts[0]
    for key in first_dict.keys():
        stacked = torch.stack([client[key].float() for client in client_dicts])
        super_dict[key] = torch.mean(stacked, dim=0)

    super_model.load_state_dict(super_dict)
    super_model.eval()
    is_model_ready = True
    st.sidebar.balloons()
else:
    st.sidebar.warning("⚠️ Upload at least 2 `.pt` files to construct the Super Model.")

# Dashboard Layout
tab1, tab2 = st.tabs(["📊 Federated Analytics Dashboard", "🔮 Super Model Prediction System"])

with tab1:
    st.header("Comparative Metadata & Predictive Analytics")
    if is_model_ready:
        col1, col2 = st.columns(2)

        # Synthetic analysis charts based on aggregated metrics
        with col1:
            st.subheader("Feature Weight Distribution across Nodes")
            weights_first_layer = super_model.net[0].weight.detach().numpy()
            avg_importance = np.mean(np.abs(weights_first_layer), axis=0)
            features = ["GRE Score", "TOEFL Score", "CGPA / GPA", "Projects", "Internships"]

            fig_importance = px.bar(
                x=features, y=avg_importance,
                labels={'x': 'Feature', 'y': 'Synthesized Weight Magnitude'},
                title="Super Model Feature Importance Matrix",
                color=avg_importance, color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_importance, use_container_width=True)

        with col2:
            st.subheader("Node Parameter Divergence")
            # Calculate distance between College A and College B weights
            diffs = []
            for k in client_dicts[0].keys():
                d = torch.norm(client_dicts[0][k].float() - client_dicts[1][k].float()).item()
                diffs.append(d)

            fig_diff = px.line(
                x=list(client_dicts[0].keys()), y=diffs,
                labels={'x': 'Neural Network Layers', 'y': 'Euclidean Distance'},
                title="Cross-College Model Discrepancy Matrix",
                markers=True
            )
            st.plotly_chart(fig_diff, use_container_width=True)
    else:
        st.info("Upload metadata files via the sidebar to unlock cross-college analytics.")

with tab2:
    st.header("Real-Time Student Placement Prediction")
    if is_model_ready:
        st.markdown("Provide candidate parameters to query the synthesized Super Model:")

        c1, c2, c3 = st.columns(3)
        with c1:
            gre = st.slider("GRE Score", 260, 340, 315)
            toefl = st.slider("TOEFL Score", 80, 120, 105)
        with c2:
            gpa = st.slider("CGPA (Scale of 10)", 5.0, 10.0, 8.5)
            projects = st.number_input("Major Projects Completed", 0, 10, 2)
        with c3:
            internships = st.number_input("Internships Completed", 0, 5, 1)

        # Standardize inputs to match model scales
        norm_gre = (gre - 260) / (340 - 260)
        norm_toefl = (toefl - 80) / (120 - 80)
        norm_gpa = (gpa - 5.0) / (10.0 - 5.0)
        norm_proj = projects / 10.0
        norm_intern = internships / 5.0

        input_tensor = torch.tensor([[norm_gre, norm_toefl, norm_gpa, norm_proj, norm_intern]], dtype=torch.float32)

        if st.button("Predict Placement Probability", type="primary"):
            with torch.no_grad():
                prob = super_model(input_tensor).item()

            st.divider()
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric(label="Super Model Placement Probability", value=f"{prob * 100:.2f}%")
            with res_col2:
                if prob >= 0.60:
                    st.success("Result: HIGH PROBABILITY OF PLACEMENT")
                elif prob >= 0.40:
                    st.warning("Result: MODERATE PROBABILITY OF PLACEMENT")
                else:
                    st.error("Result: LOW PROBABILITY OF PLACEMENT")
    else:
        st.warning("Please upload metadata files to activate the prediction engine.")