import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Load the pre-trained model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("best_model_mlp_OD.pkl")  # Replace with your actual model file
    scaler = joblib.load("scaler.pkl")  # Replace with your actual scaler file
    return model, scaler

# Function to generate compact circular grids
def generate_grid(diameter, layer_index, total_layers):
    radius = diameter / 120 / 2  # Diameter to radius in grid blocks
    grid_blocks = []
    for x in range(-int(radius), int(radius) + 1):
        for z in range(-int(radius), int(radius) + 1):
            if x**2 + z**2 <= radius**2:
                grid_blocks.append({
                    "x": x,
                    "y": (total_layers - layer_index - 1) * 1,  # Top layer has highest y-value
                    "z": z
                })
    return grid_blocks

# Add custom CSS for font sizes
st.markdown(
    """
    <style>
    /* Adjust font size of sidebar labels, words, and figures */
    .css-1v3fvcr, .css-16huue1, .css-qrbaxs {
        font-size: 18px !important;
    }
    /* Adjust main title font size */
    .css-10trblm {
        font-size: 36px !important; /* Main header */
    }
    /* Adjust subheader font size */
    .css-1w2kcmx {
        font-size: 20px !important; /* Subheaders */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit UI
st.title("CO2 Plume Shape Evolution")

# Sidebar for inputs
st.sidebar.header("Model Inputs")
st.sidebar.write("Adjust the sliders to set input values for prediction.")

# Define input sliders
inputs = {}
inputs["Injection_Rate_MMT"] = st.sidebar.slider("Injection Rate (MMT)", min_value=1.0, max_value=6.0, step=0.1, value=2.0)
inputs["PERMI"] = st.sidebar.slider("Horizontal Permeability (mD)", min_value=100, max_value=600, step=10, value=100)
inputs["PERMK"] = st.sidebar.slider("Vertical Permeability (mD)", min_value=10, max_value=60, step=10, value=10)
inputs["Porosity"] = st.sidebar.slider("Porosity", min_value=0.1, max_value=0.35, step=0.01, value=0.1)
inputs["Initial_Pressure"] = st.sidebar.slider("Initial Pressure (KPa)", min_value=13000, max_value=24000, step=1000, value=16000)
inputs["Temperature"] = st.sidebar.slider("Temperature (°C)", min_value=40, max_value=80, step=5, value=40)
inputs["Reservoir_Thickness"] = st.sidebar.slider("Reservoir Thickness (m)", min_value=310, max_value=600, step=10, value=310)
inputs["SGR"] = st.sidebar.slider("Resdual Gas Saturation", min_value=0.2, max_value=0.35, step=0.01, value=0.3)
inputs["Rock_Compressibility"] = st.sidebar.slider("Rock Compressibility (1/KPa)", min_value=4e-7, max_value=4e-5, step=1e-6, value=4e-6, format="%.6f")
inputs["Salinity"] = st.sidebar.slider("Salinity (ppm)", min_value=20000, max_value=40000, step=1000, value=20000)

# Add slider for selecting year
start_year = 2024
timesteps_per_layer = 81
inputs["Selected_Year"] = st.sidebar.slider("Select Year", min_value=start_year, max_value=start_year + timesteps_per_layer - 1, step=1)

# Convert inputs to DataFrame
selected_year = inputs.pop("Selected_Year")  # Separate the year selection
input_df = pd.DataFrame([inputs])

# Load model and scaler
model, scaler = load_model_and_scaler()

# Scale inputs
scaled_inputs = scaler.transform(input_df)

# Make predictions
predictions = model.predict(scaled_inputs)

# Process predictions
predictions = np.clip(predictions, a_min=0, a_max=None)  # Ensure non-negative values
predictions = np.round(predictions)  # Round predictions to the nearest integer

# Scale predictions to diameters for visualization
predicted_diameters = predictions[0] * 120  # Scale predictions
layers = np.split(predicted_diameters, 6)  # Split into 6 layers (81 timesteps each)

# Calculate the timestep index
timestep_index = selected_year - start_year

# Plot the temporal graph
st.subheader("2D Temporal Evolution of Plume Diameters")
fig, ax = plt.subplots(figsize=(10, 5))

# Plot data for each layer
years = np.arange(start_year, start_year + timesteps_per_layer)
for layer_index, layer_data in enumerate(layers):
    ax.plot(years, layer_data, label=f"Layer {layer_index + 1}")

# Graph styling
ax.set_xlim(start_year, start_year + timesteps_per_layer - 1)
ax.set_ylim(0, 7500)
ax.set_xlabel("Year", fontsize=14)
ax.set_ylabel("Plume Diameter (meters)", fontsize=14)
ax.legend(fontsize=12)
ax.grid(True)

# Display the temporal graph
st.pyplot(fig)

# Create layout with two columns: 3D Plume and Plume Diameter Values
col1, col2 = st.columns([3, 1])

# Column 1: Display 3D plume
with col1:
    st.subheader(f"3D Plume Distribution for Year {selected_year}")
    fig = go.Figure()

    # Generate 3D plot for the selected timestep
    for layer_index, layer_data in enumerate(layers):
        diameter = layer_data[timestep_index]
        grid_blocks = generate_grid(diameter, layer_index, len(layers))
        for block in grid_blocks:
            fig.add_trace(go.Scatter3d(
                x=[block["x"]],
                y=[block["y"]],
                z=[block["z"]],
                mode='markers',
                marker=dict(
                    size=5,
                    color="red" if layer_index == 0 else "green",
                    opacity=1.0
                ),
                showlegend=False
            ))

    # Add rotation animation for 5 complete rotations
    angles = np.linspace(0, 360 * 5, 75)  # 75 frames for 5 rotations
    frames = [
        go.Frame(
            layout=dict(
                scene_camera=dict(
                    eye=dict(
                        x=2 * np.sin(np.radians(angle)),
                        y=2 * np.cos(np.radians(angle)),
                        z=1.25,
                    )
                )
            )
        )
        for angle in angles
    ]

    fig.frames = frames

    # Add a Rotate button that starts from the current angle and rotates 5 times
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=True,
                buttons=[
                    dict(
                        label="Rotate",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=50, redraw=True),
                                fromcurrent=True,  # Starts rotation from the current angle
                                loop=False,  # Ensures it stops after 5 rotations
                            ),
                        ],
                    )
                ],
            )
        ],
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )

    # Display the 3D plot
    st.plotly_chart(fig, use_container_width=True)

# Column 2: Display plume diameter values
with col2:
    st.write("### Plume Diameter Values (meters)")
    for layer_index, layer_data in enumerate(layers):
        diameter = layer_data[timestep_index]
        st.write(f"Layer {layer_index + 1}: **{diameter:.2f} meters**")
