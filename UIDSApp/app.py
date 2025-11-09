import streamlit as st
import os
from EvaluateOnnx import EvaluateModelOnnx

# ======================================================
# CONFIGURATION
# ======================================================
MODEL_PATHS = {
    "Kia": "/home/lisa/Arupreza/UIDS-II/UIDSApp/OnnxModels/TrainKiaOnnx",
    "Genesis": "/home/lisa/Arupreza/UIDS-II/UIDSApp/OnnxModels/TrainGenOnnx",
    "Tesla": "/home/lisa/Arupreza/UIDS-II/UIDSApp/OnnxModels/TrainTeslaOnnx",
    "Silverado": "/home/lisa/Arupreza/UIDS-II/UIDSApp/OnnxModels/TrainSilOnnx",
}

# ======================================================
# PAGE SETUP
# ======================================================
st.set_page_config(page_title="UIDS ONNX Dashboard", layout="centered")
st.title("🚗 UIDS ONNX Model Validation & Inference Dashboard")

st.markdown("""
Select a **vehicle model**, configure parameters, and click **Start**  
to run either **validation (with labels)** or **real-life inference (no labels)** on your CAN dataset.
""")

# ======================================================
# SESSION STATE INITIALIZATION
# ======================================================
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "model_path" not in st.session_state:
    st.session_state.model_path = None
if "mode" not in st.session_state:
    st.session_state.mode = "validation"

# ======================================================
# MODEL SELECTION
# ======================================================
st.subheader("1️⃣ Select Vehicle Model")
cols = st.columns(len(MODEL_PATHS))
for i, (name, path) in enumerate(MODEL_PATHS.items()):
    with cols[i]:
        if st.button(f"🧠 {name}"):
            st.session_state.selected_model = name
            st.session_state.model_path = path

if not st.session_state.selected_model:
    st.info("👉 Please select a model to start.")
    st.stop()

model_name = st.session_state.selected_model
model_path = st.session_state.model_path
st.success(f"✅ Selected model: **{model_name}**")
st.caption(f"📁 {model_path}")

# ======================================================
# CONFIGURE PARAMETERS
# ======================================================
st.subheader("2️⃣ Configure Run Parameters")

data_dir = st.text_input(
    "📂 Path to CAN CSV Folder:",
    "/home/lisa/Arupreza/UIDS-II/Split_data/Test/Tesla/Lower Low"
)

time_gap = st.number_input(
    "⏱ Time Gap (seconds)", min_value=1.0, max_value=500.0, value=83.0, step=1.0
)

device = st.selectbox("💻 ONNX Runtime Device", ["cpu", "cuda"])

mode = st.radio(
    "⚙️ Select Mode",
    ["Validation (with Labels)", "Real-Life Inference (no Labels)"],
    horizontal=True,
    index=0 if st.session_state.mode == "validation" else 1,
)

# Keep user’s selection persistent
st.session_state.mode = "validation" if "Validation" in mode else "inference"

# ======================================================
# RUN BUTTON
# ======================================================
st.subheader("3️⃣ Run Evaluation or Inference")

if st.button("🚀 Start"):
    with st.spinner(f"Running {st.session_state.mode.upper()} on {model_name} model..."):
        try:
            result = EvaluateModelOnnx(
                model_path=model_path,
                data_directory=data_dir,
                time_gap=time_gap,
                mode=st.session_state.mode,
                device=device
            )

            if st.session_state.mode == "validation":
                st.success(f"✅ Validation Completed for {model_name}")
                st.write(f"**Accuracy:** {result['accuracy']:.4f}")
                st.write(f"**F1 Score:** {result['f1']:.4f}")
                st.write(f"**Precision:** {result['precision']:.4f}")
                st.write(f"**Recall:** {result['recall']:.4f}")

            if "cm" in result:
                import matplotlib.pyplot as plt
                import numpy as np

                cm = result["cm"]
                cm_sum = cm.sum(axis=1, keepdims=True)
                cm_perc = np.round((cm / cm_sum) * 100, 2)

                # Combined text: count + percentage
                labels = np.array([
                    [f"{cm[0,0]}\n{cm_perc[0,0]:.2f}%", f"{cm[0,1]}\n{cm_perc[0,1]:.2f}%"],
                    [f"{cm[1,0]}\n{cm_perc[1,0]:.2f}%", f"{cm[1,1]}\n{cm_perc[1,1]:.2f}%"]
                ])

                fig, ax = plt.subplots(figsize=(5, 5))
                im = ax.imshow(cm, cmap="BuPu", interpolation="nearest")

                # Titles and labels
                ax.set_title(f"Confusion Matrix - {model_name}", fontsize=16, fontweight='bold', pad=20)
                ax.set_xlabel("True Labels", fontsize=13, labelpad=10, color="navy")
                ax.set_ylabel("Predicted Labels", fontsize=13, labelpad=10, color="blue")

                # Axis ticks
                ax.set_xticks(np.arange(2))
                ax.set_yticks(np.arange(2))
                ax.set_xticklabels(["Attack Free", "Attack"], fontsize=12, color="navy")
                ax.set_yticklabels(["Attack Free", "Attack"], fontsize=12, color="green")

                # Grid for clarity instead of black rectangles
                ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
                ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
                ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
                ax.tick_params(which="minor", bottom=False, left=False)

                # Annotate counts + percentages
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, labels[i, j],
                                ha="center", va="center", fontsize=13,
                                color="white" if cm_perc[i, j] > 50 else "black",
                                fontweight="bold")

                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                plt.tight_layout()
                st.pyplot(fig)


            else:
                st.success(f"✅ Real-Life Inference Completed for {model_name}")
                st.write(f"**Total Segments Processed:** {result.get('segments', 'N/A')}")
                if "predictions" in result:
                    st.write("**Sample Predictions:**")
                    st.json(result["predictions"][:15])

        except Exception as e:
            st.error("❌ Error during execution:")
            st.exception(e)