from langchain_groq import ChatGroq
from ingestion import get_column_info, retrive_releavant_data
import json
import pandas as pd
import re
from quickchart import QuickChart
qc = QuickChart()

# LLM Instances
sql_llm = ChatGroq(
    temperature=0,
    model_name='qwen-2.5-coder-32b',
    groq_api_key="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"
)


planner_llm = ChatGroq(
    temperature=0,
    model_name='Llama-3.3-70b-Versatile',
    groq_api_key="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"
)



data_llm = ChatGroq(
    temperature=0,
    model_name='Llama-3.3-70b-Versatile',
    groq_api_key="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"
)


def planner(user_prompt):
    """
    Uses LLM to generate structured SQL-specific prompts and visualization prompts if applicable.
    """
    planning_prompt = f"""
        ### Instructions:
        - Ensure the output is **only** the JSON object, without any extra text.

        ### Task:
        - Decompose the user request into **separate SQL-specific prompts**.
        - Each prompt should be **a single query** targeting specific data.
        - Ensure clarity in each prompt for accurate SQL generation.
        - Identify if the request explicitly mentions visualization (e.g., "show", "plot", "graph", "chart", "trend", "distribution", "pattern", "visualize").
        - If the request does not mention visualization but involves comparing trends over time or distributions, include a visualization key.
        - If the request is purely numerical or textual data retrieval, do NOT generate a visualization.

        ### Example:
        **User Request:** "Compare the average revenues of GOOG and AAPL from 2021 to 2023 and show it in a bar chart."

        **Generated Output:**
        {{
            "prompts": [
                "Get the average revenue of AAPL from 2021 to 2023.",
                "Get the average revenue of GOOG from 2021 to 2023."
            ],
            "visualization": "Generate a bar chart comparing the average revenues of GOOG and AAPL from 2021 to 2023. Ensure axis labels and units are clearly defined."
        }}

        ### User Request:
        {user_prompt}
    """

    response = planner_llm.invoke(planning_prompt).content.strip()
    print(response)

    try:
        parsed_response = json.loads(response)
        sql_prompts = parsed_response.get("prompts", [])
        visualization_prompt = parsed_response.get("visualization", None)
    except json.JSONDecodeError:
        sql_prompts = [user_prompt]  # Fall back if LLM output is incorrect
        visualization_prompt = None
    
    return sql_prompts, visualization_prompt


def generate_sql_query(sub_prompt):
    """
    Converts an SQL prompt into an actual SQL query.
    """
    column_info,categorical_values= get_column_info()
    column_str = "\n".join([f"{col}: {dtype}" for col, dtype in column_info.items()])

    llm_prompt = f"""
        Instruction:

        - Return only the SQL query, nothing else.
        - The table name is **data** and must be written **plainly** without backticks (` `).
        - Do NOT include any quotations around the SQL query; return it as a plain string.
        - Represent **column names only** in SQL format using backticks (`` `column_name` ``).
        - Ensure the SQL syntax is valid for querying a Pandas DataFrame with SQLite.
        - Design the query to be comprehensive and context-rich, including all columns necessary to fully address the user’s request, considering that the results will be summarized by another LLM.
        - Infer the user’s intent intelligently: include relevant dimensions (e.g., time periods, identifiers) and metrics that provide a complete picture, even if not explicitly requested, unless it contradicts the user’s intent.
        - Avoid overly generic or incomplete queries; prioritize clarity and utility for downstream analysis and summarization.
        - Use precise conditions to filter data based on the user’s request (e.g., exact ticker, year range, categories), and order results logically (e.g., by year) when applicable.

        Database Schema:
        {column_str}

        Categorical Mappings:
        {json.dumps(categorical_values, indent=2)}

        User Request:
        {sub_prompt}
    """

    response = sql_llm.invoke(llm_prompt).content.strip()
    return response


def execute_queries(sql_prompts):
    """
    Executes multiple SQL queries and merges results.
    """
    combined_df = pd.DataFrame()
    
    for prompt in sql_prompts:
        sql_query = generate_sql_query(prompt)
        print(sql_query)
        result_df = retrive_releavant_data(sql_query)
        
        if not result_df.empty:
            result_df["query_step"] = prompt  # Tag data with its step
            combined_df = pd.concat([combined_df, result_df], ignore_index=True)

    return combined_df


def summarize_data_result(result_df, user_prompt):
    """
    Summarizes the retrieved SQL data.
    """
    if result_df.empty:
        return "No relevant data found."

    result_json = result_df.to_json()

    summary_prompt = f"""
        ### Task:
        - Summarize the extracted data for the given user request.
        - Ensure clarity, conciseness, and relevance.
        - Highlight key insights and trends.

        ### User Request:
        {user_prompt}

        ### Extracted Data:
        {result_json}
    """

    response = data_llm.invoke(summary_prompt).content.strip()
    return response


def summarize_data_result(result_df,user_prompt):
    """
    - Converts DataFrame to JSON.
    - Passes it to `data_llm` for natural language summarization.
    """
    result_json = result_df.to_json()

    summary_prompt = f"""
        ### Instructions:
        - You are an expert data analyst.
        - Based on the given user request and JSON data, generate a **concise and insightful summary**.
        - The response should be in **natural language**, avoiding technical jargon while highlighting key insights.
        - Ensure the summary is **relevant to the user's original request**.

        ### User Request:
        {user_prompt}

        ### Data:
        {result_json}
    """


    response = data_llm.invoke(summary_prompt).content.strip()
    return response


def visusalize(result_df , user_prompt):
    """
    Converts data frame into a chart.js for more accurate reading and undersatding 
    """

    result_json = result_df.to_json()
    prompt = f"""

    ### Instructions:
     Ensure the output is **only** the JSON  without any extra text. Do not include any backticks or json written just return the json 
     Follows JSON standard syntax, which is compatible with APIs like QuickChart.io.
     Uses double quotes (") for keys and strings, which is  valid JSON.
     Do NOT include explanations, comments, or extra text—just return the JSON.
    I want you to act as a Chart is visualization expert. Your primary goal is to generate valid and clearly labeled Chart. js configuration objects based on user requests. These configurations should-be directly usable within a Chart.js environment.
    Input: You will receive user requests describing the desired visualization. These requests may include:
    Chart Type: (e-g-, bar, line, pie, scatten, radar, bubble, etc.)
    Data: The data to be visualized. This can be provided in various formats (CSV, JSON, lists) .
    Clarify ambiguous data formats with the user. Handle data extraction as needed.
    Labels: Labels for data points, axes, and other elements. Specifically, the user must provide clear axis titles and units (if applicable). Ask clarifying questions if this information is missing.
    Styling: Customization options (colors, fonts, gridlines, titles, legends, tooltips, axis scales). Use Chart.js defaults if no specific styling is requested.
    Specific Chart-js options: Users might request specific Chart.js features (logarithmic scales, animations, plugins).
    Natural language descriptions: Interpret less structured descriptions and translate them into valid Chart-js configurations.
    Axis information: Request and expect clear details on axis labels, including:
    Axis Titles: Concise titles for x and y axes. (e-g-, "Month", "Sales (USD)")
    Units: Units of measurement, if relevant. (e-g., "(USD)", "kg")
    Data Type: The type of data on each axis (categorical, numerical, time series).
    Specific Formatting: Requirements for date/number formats, currency symbols, etc.

    Output: A valid, well-formatted, and easily readable JSON object representing a Chart.js configuration, ready to be used in a new Chart() constructor. The output must include properly configured axis labels based on the provided input. This includes adding titles and units to both axes within the options.scales section of the JSON.

    The response must:
    Follow JSON standard syntax.
    Be compatible with APIs like QuickChart.io.
    Not include any backticks or explicitly written JSON.

    ### User Request:
    {user_prompt}

    ### Data:
    {result_json}

    """
    response = data_llm.invoke(prompt).content.strip()
    cleaned_response = re.sub(r"^```json\n|\n```$", "", response).strip()
    print("*****************************")
    print(cleaned_response)

    try:
        result = json.loads(cleaned_response)
        qc.config = result

        
    except:
        result = {}
        qc.config = result
        print("not possible")
    return qc.get_url()



def process_user_query(user_prompt):
    """
    Full pipeline execution:
    1. Plans SQL-specific prompts.
    2. Generates and executes SQL queries.
    3. Summarizes the retrieved data.
    """
    sql_prompts,visualisation_prompt = planner(user_prompt)
    # print(sql_prompts)
    combined_df = execute_queries(sql_prompts)
    # print(combined_df)
    summary = summarize_data_result(combined_df, user_prompt)
    # print(summary)
    if visualisation_prompt:
        visualisation = visusalize(combined_df,visualisation_prompt)
        # print(visualisation)
    else:
        visualisation = None


    return combined_df, summary ,visualisation





# print(generate_sql_query("Compare EPS of google and microsoft"))


import streamlit as st
import pandas as pd
from datetime import datetime

# Importing necessary functions from your pipeline
from ingestion import get_column_info, retrive_releavant_data
from langchain_groq import ChatGroq
import json


# Streamlit Chat UI
st.set_page_config(page_title="Chat with Data", layout="wide")
st.title("📊 Data Assistant Chat")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display previous messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if user_input := st.chat_input("Ask something about the data..."):
    # Display user message
    st.chat_message("user").markdown(user_input)
    st.session_state["messages"].append({"role": "user", "content": user_input})
    
    # Process user query
    _, response,visualisation = process_user_query(user_input)
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
        st.image(visualisation)
    
    # Save response to chat history
    st.session_state["messages"].append({"role": "assistant", "content": response})
