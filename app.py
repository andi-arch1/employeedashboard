import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

ollama_url = "http://localhost:11434/"

# Streamlit page configuration
st.set_page_config(layout="wide")

# Define the AI prompt template
template = """You are an AI assistant that understands datasets.
Here is the conversation history: {context}
Here is the dataset summary: {dataset_summary}
Question: {question}
Answer: """

# Initialize the AI model and prompt
model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
Chain = prompt | model

# Sidebar for file upload
st.sidebar.title("Upload CSV for Analysis")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"], key="file_uploader")

# Session state for context and dataset summary
if "context" not in st.session_state:
    st.session_state.context = ""
if "dataset_summary" not in st.session_state:
    st.session_state.dataset_summary = "No dataset uploaded yet."

if uploaded_file:
    # Read the uploaded file into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Validate required columns for HCCBD analysis
    required_columns = [
        "Year", "Campus", "Month", "Source of Channel", "Position", "Gender", "Age", "GPA"
    ]
    if set(required_columns).issubset(df.columns):
        st.sidebar.success("Dataset validated for HCCBD analysis.")

        # Filters for HCCBD analysis
        st.header("HCCBD Analysis")
        selected_campus = st.multiselect("Select Campus", options=df["Campus"].unique(), default=df["Campus"].unique())
        selected_month = st.multiselect("Select Month", options=df["Month"].unique(), default=df["Month"].unique())
        selected_channel = st.multiselect("Select Source of Channel", options=df["Source of Channel"].unique(), default=df["Source of Channel"].unique())
        selected_position = st.multiselect("Select Position", options=df["Position"].unique(), default=df["Position"].unique())
        selected_gender = st.multiselect("Select Gender", options=df["Gender"].unique(), default=df["Gender"].unique())
        gpa_filter = st.slider("Minimum GPA", min_value=2.5, max_value=4.0, value=3.2, step=0.1)

        # Apply filters
        filtered_df = df[
            (df["Campus"].isin(selected_campus)) &
            (df["Month"].isin(selected_month)) &
            (df["Source of Channel"].isin(selected_channel)) &
            (df["Position"].isin(selected_position)) &
            (df["Gender"].isin(selected_gender)) &
            (df["GPA"] >= gpa_filter)
        ]

        # Split data by year for comparison
        current_year = 2024
        last_year = 2023
        df_current_year = filtered_df[filtered_df["Year"] == current_year]
        df_last_year = filtered_df[filtered_df["Year"] == last_year]

        # Display data counts
        st.subheader("Data Overview")
        st.write(f"Total records for {current_year}: {len(df_current_year)}")
        st.write(f"Total records for {last_year}: {len(df_last_year)}")

        # Visualization function
        def plot_comparison(group_by_column):
            current_year_count = df_current_year[group_by_column].value_counts().reset_index()
            current_year_count.columns = [group_by_column, "Count"]
            current_year_count["Year"] = "2024"

            last_year_count = df_last_year[group_by_column].value_counts().reset_index()
            last_year_count.columns = [group_by_column, "Count"]
            last_year_count["Year"] = "2023"

            combined = pd.concat([current_year_count, last_year_count])

            fig = px.bar(
                combined,
                x=group_by_column,
                y="Count",
                color="Year",
                barmode="group",
                title=f"Comparison by {group_by_column}"
            )
            st.plotly_chart(fig)

        # Comparison button
        st.subheader("Comparison: This Year vs Last Year")
        comparison_column = st.selectbox("Select a column to compare:", ["Campus", "Source of Channel", "Position", "Gender"])
        plot_comparison(comparison_column)

        # Display filtered data
        st.subheader("Filtered Data")
        st.dataframe(filtered_df)

    # Generate dataset summary for AI
    numerical_stats = df.select_dtypes(include="number").describe().to_dict()
    categorical_stats = {
        col: {
            "unique_values": df[col].nunique(),
            "most_frequent": df[col].mode()[0] if not df[col].mode().empty else None,
        }
        for col in df.select_dtypes(include="object").columns
    }

    dataset_summary = f"Numerical Statistics: {numerical_stats}\n\nCategorical Statistics: {categorical_stats}"
    st.session_state.dataset_summary = dataset_summary

    # Display the dataset preview
    st.header("AI-Powered Chat")
    # Input box for user question
    user_input = st.text_input("Ask a question about the dataset:", key="user_input")

    # Submit button
    generate_button = st.button("Send")

    if generate_button and user_input:
        # Generate the bot's response
        result = Chain.invoke({
            "context": st.session_state.context,
            "dataset_summary": st.session_state.dataset_summary,
            "question": user_input,
        })

        # Update the context with user and AI conversation
        st.session_state.context += f"\nUser: {user_input}\nAI: {result}"

        # Display the conversation
        st.write(f"**You:** {user_input}")
        st.write(f"**Bot:** {result}")

else:
    st.sidebar.warning("Please upload a CSV file to proceed.")
