import pandas as pd
import altair as alt
import streamlit as st
# from local_components import card_container
import matplotlib.pyplot as plt

# Example DataFrame (Replace this with your real data)
data = {
    "employee_id": [1, 2, 3, 4, 5],
    "department": ["HR", "IT", "Finance", "IT", "HR"],
    "region": ["North", "South", "East", "North", "West"],
    "education": ["Bachelor's", "Master's", "PhD", "Bachelor's", "Master's"],
    "gender": ["Male", "Female", "Male", "Female", "Male"],
    "recruitment_channel": ["Referral", "Online", "Agency", "Online", "Referral"],
    "no_of_trainings": [2, 5, 3, 4, 1],
    "age": [25, 35, 45, 30, 40],
    "previous_year_rating": [4, 3, 5, 2, 4],
    "length_of_service": [2, 10, 7, 5, 8],
    "awards_won": [0, 1, 0, 1, 0],
    "avg_training_score": [80, 90, 85, 70, 88]
}
df = pd.DataFrame(data)

st.set_page_config(layout="wide")  # Add this at the start of your script

# Streamlit App
st.title("Employee Analytics Dashboard")

# Create three sections for filtering, metrics, and visualizations
col1, col2, col3 = st.columns([1.5, 2, 2.5])  # Adjust column width ratios

# Left Column (col1): Filters and Inputs
with col1:
    st.header("Filters")
    
    # Filter by Department
    departments = st.multiselect(
        "Select Department",
        options=df["department"].unique(),
        default=df["department"].unique()
    )
    
    # Filter by Region
    regions = st.multiselect(
        "Select Region",
        options=df["region"].unique(),
        default=df["region"].unique()
    )
    
    # Filter by Gender
    genders = st.multiselect(
        "Select Gender",
        options=df["gender"].unique(),
        default=df["gender"].unique()
    )
    
    # Filter by Age Range
    age_range = st.slider(
        "Select Age Range",
        min_value=int(df["age"].min()),
        max_value=int(df["age"].max()),
        value=(int(df["age"].min()), int(df["age"].max()))
    )
    
    # Filter the DataFrame based on selections
    filtered_df = df[
        (df["department"].isin(departments)) &
        (df["region"].isin(regions)) &
        (df["gender"].isin(genders)) &
        (df["age"] >= age_range[0]) &
        (df["age"] <= age_range[1])
    ]

# Middle Column (col2): Key Metrics
with col2:
    st.header("Key Metrics")
    if not filtered_df.empty:
        # Display metrics
        col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
        col_metrics1.metric(label="Average Age", value=f"{filtered_df['age'].mean():.2f}")
        col_metrics2.metric(label="Avg Training Score", value=f"{filtered_df['avg_training_score'].mean():.2f}")
        col_metrics3.metric(label="Total Awards Won", value=filtered_df["awards_won"].sum())
    else:
        st.write("No data available for the selected filters.")

# Right Column (col3): Visualizations
with col3:
    st.header("Visualizations")
    
    # Bar Chart: Average Training Score by Department
    st.subheader("Avg Training Score by Dept")
    if not filtered_df.empty:
        avg_training_by_dept = filtered_df.groupby("department")["avg_training_score"].mean()
        st.bar_chart(avg_training_by_dept)
    else:
        st.write("No data available for the selected filters.")
    
    # Pie Chart: Gender Distribution
    st.subheader("Gender Distribution")
    if not filtered_df.empty:
        gender_counts = filtered_df["gender"].value_counts()
    
        # Create a Plotly pie chart
        fig = px.pie(gender_counts, values=gender_counts.values, names=gender_counts.index, hole=.3)
        fig.update_layout(title_text="Gender Distribution")
    
        # Pass the figure to st.plotly_chart()
        st.plotly_chart(fig)
    else:
        st.write("No data available for the selected filters.")
