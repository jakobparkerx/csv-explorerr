import streamlit as st
import pandas as pd
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) # loads api key from .env file

def detect_sensitive_columns(file):
# defines keywords that are considered sensitive information and checks if they are present
    sensitive_keywords = [
        "name", "email", "phone", "address", "account number"
    ]
    detected = []
    for col in file.columns:
        if any(keyword in col.lower() for keyword in sensitive_keywords):
            detected.append(col)
    return detected

def remove_sensitive_columns(file, sensitive_cols):
    # removes sensitive columns
    return file.drop(columns=sensitive_cols)


def summarize_data(file):
    # summarises data for analysis
    summary = {
        "num_rows": len(file),
        "columns": {}
    }
    for col in file.columns:
        if pd.api.types.is_numeric_dtype(file[col]):
            # if column is numeric, calculates statistics
            summary["columns"][col] = {
                "type": "numeric",
                "mean": file[col].mean(),
                "median": file[col].median(),
                "min": file[col].min(),
                "max": file[col].max(),
                "range": file[col].max() - file[col].min(),
            }
        else:
            summary["columns"][col] = {
                #if column is non-numeric, summarises as categorical
                "type": "categorical",
                "unique_values": file[col].nunique(),
                "top_values": file[col].value_counts().head(3).to_dict(),
            }
    return summary


def prepare_prompt(summary):
    return json.dumps(summary, indent=2)

def ask_chatgpt(prompt_text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful data analyst."},
            {"role": "user", "content": prompt_text}
        ]
    )
    return response.choices[0].message.content



def main():

    col1, col2, col3 = st.columns([1, 20, 1])
    with col2:
        st.markdown(
            "<h1 style='text-align: center;'>📊 CSV Column Cleaner</h1>",
            unsafe_allow_html=True
        )
    st.info("A simple tool to clean and analyze your CSV files. Upload a CSV file, select a column, and view its statistics. You can also edit the data directly in the app, and then redownload it as a CSV.")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        file = pd.read_csv(uploaded_file)
        cols = st.columns(len(file.columns))  # column list

        st.subheader("Original Data")

        selected_column = st.selectbox("Columns:", file.columns)

        # work out stats to display
        if selected_column:
            if pd.api.types.is_numeric_dtype(file[selected_column]):
                stats = {
                    "Row Total": file.shape[0],
                    "Mean": file[selected_column].mean(),
                    "Median": file[selected_column].median(),
                    "Min": file[selected_column].min(),
                    "Max": file[selected_column].max(),
                    "Range": file[selected_column].max() - file[selected_column].min(),
                    "Missing Values": file[selected_column].isnull().sum(),
                    "Percentage Missing": (file[selected_column].isnull().sum() / file.shape[0]) * 100, #need to make this an int?
                }
                selected_ranges = st.multiselect("Statistics:", stats.keys())
                if selected_ranges != []:
                    st.write(f"Summary for {selected_column}:")

                cols = st.columns(2)
                for i, stat in enumerate(selected_ranges):
                    with cols[i % 2]:
                        st.info(f"{stat.capitalize()}: {stats[stat]}")


            else:
                stats = {
                    "Row Total": file.shape[0],
                    "Unique Values": file[selected_column].nunique(),
                    "Top Values": file[selected_column].value_counts().head(3).to_dict(),
                    "Missing Values": file[selected_column].isnull().sum(),
                    "Percentage Missing": (file[selected_column].isnull().sum() / file.shape[0]) * 100,
                    "Percentage Unique": (file[selected_column].nunique() / file.shape[0]) * 100
                }
                selected_ranges = st.multiselect("Statistics:", stats.keys())
                if selected_ranges != []:
                    st.write(f"Summary for {selected_column}:")

                cols = st.columns(2)
                for i, stat in enumerate(selected_ranges):
                    with cols[i % 2]:
                        st.info(f"{stat.capitalize()}: {stats[stat]}")

        st.subheader("Data Editor")

        show_missing_only = st.checkbox("Show only rows with missing values")

        if show_missing_only:
            file_to_edit = file[file.isnull().any(axis=1)]
        else:
            file_to_edit = file

        st.data_editor(file_to_edit, use_container_width=True, hide_index=True)

        st.download_button(
            label="Download cleaned CSV",
            data=file_to_edit.to_csv(index=False),
            file_name="cleaned_data.csv",
            mime="text/csv"
        )

        if st.button("Get AI Analysis"):
            with st.spinner("Analyzing data summary with ChatGPT..."):
                try:
                    summary = summarize_data(file)
                    prompt = prepare_prompt(summary)
                    analysis = ask_chatgpt(f"Analyze this data summary and provide insights:\n{prompt}")
                    st.subheader("ChatGPT Analysis")
                    st.write(analysis)
                except Exception as e:
                    st.error(f"Error calling OpenAI API: {e}")


if __name__ == "__main__":
    main()








