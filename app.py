from langchain_groq import ChatGroq
from ingestion import get_column_info, retrive_releavant_data , get_columns,get_dataframes
import json
import pandas as pd
import re
from quickchart import QuickChart
import os
from  openai import OpenAI

qc = QuickChart()


global dataframes 
dataframes = get_dataframes()


gclient = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key= "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"
)


# LLM Instances
sql_llm = ChatGroq(
    temperature=0,
    model_name='qwen-2.5-coder-32b',
    groq_api_key="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"
)


planner_llm = ChatGroq(
    temperature=0.3,
    # model_name='Llama-3.3-70b-Versatile',
    model_name = 'llama3-70b-8192',
    groq_api_key="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"
)



data_llm = ChatGroq(
    temperature=0.3,
    # model_name='Llama-3.3-70b-Versatile',
    model_name = 'llama3-70b-8192',
    groq_api_key="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"
)









def get_tables_info(folder_path=r'C:\Desktop\experiments\data_analytics\database'):
    tables = {}
    
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file).replace("\\", "/")  # Normalize path
            tables[os.path.splitext(file)[0]] = get_columns(file_path)
    
    return tables






def planner(user_prompt):
    """
    Uses LLM to identify relevant tables and generate structured SQL-specific prompts.
    Each prompt is mapped to the appropriate table(s).
    """
    tables = get_tables_info()  # Fetch table metadata

    # Convert table info into a string format for LLM context
    table_schema = json.dumps(tables, indent=4)

    planning_prompt = f"""
        ### Instructions:
        - Ensure the output is **only** the JSON object, without any extra text.
        - Use the given table schemas to determine the right tables for each request.
        - Do NOT generate SQL queries—just structured prompts.
        - Use only the table names mentioned in the Available Tables

        ### Available Tables:
        {table_schema}

        ### Task:
        - Identify the relevant table(s) for each part of the request.
        - Generate **clear SQL-specific prompts** to query the data.
        - Ensure each prompt explicitly **mentions the relevant table(s)**.
        - If a request involves multiple tables, list all necessary tables.

        ### Example:
        **User Request:** "Compare the average revenues of GOOG and AAPL from 2021 to 2023."

        **Generated Output:**
        {{
            "prompts": [
                {{
                    "prompt": "Retrieve the average revenue of AAPL for years 2021 to 2023 from the financials table.",
                    "tables": ["financials"]
                }},
                {{
                    "prompt": "Retrieve the average revenue of GOOG for years 2021 to 2023 from the financials table.",
                    "tables": ["financials"]
                }}
            ],
            "visualization": "Generate a bar chart comparing the average revenues of GOOG and AAPL from 2021 to 2023."
        }}

        ### User Request:
        {user_prompt}
    """

    # response = planner_llm.invoke(planning_prompt).content.strip()
    # print(response)  # Debugging: See LLM response

    res = gclient.chat.completions.create(
                model='llama3-70b-8192',
            messages=[   # <-- Correct spelling
                {
                    "role": "user",
                    "content": planning_prompt
                }
            ],
            temperature=0.2,
            response_format={'type': "json_object"},
        )

    response = res.choices[0].message.content

    try:
        parsed_response = json.loads(response)
        prompts = parsed_response.get("prompts", [])
        visualization_prompt = parsed_response.get("visualization", None)
    except json.JSONDecodeError:
        prompts = [{"prompt": user_prompt, "tables": []}]  # Fallback
        visualization_prompt = None
    
    return prompts, visualization_prompt





def generate_sql_query(sub_prompt, data_dict):
    """
    Converts an SQL prompt into an actual SQL query.
    """
    column_info, categorical_values = get_column_info(data_dict)

    # Format column info properly for each table
    column_str = "\n\n".join([
        f"Table: {table}\n" + "\n".join([f"  {col}: {dtype}" for col, dtype in cols.items()])
        for table, cols in column_info.items()
    ])

    llm_prompt = f"""
        Instruction:
        - Use **only the exact table names** from the provided database schema.
        - Return only the SQL query, nothing else.
        - Use **only the exact table names** from the provided database schema.
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


    # llm_prompt = f"""         
    # Instruction:          
    # - Return only the SQL query, nothing else.         
    # - In the SQL query, use "self" as the table name, do not use exact table names.
    # - Do NOT include any quotations around the SQL query; return it as a plain string.                
    # - Ensure the SQL syntax is valid for querying a polars DataFrame.

    # - IMPORTANT: Always include contextual identifier columns in your query results:
    # * Always include primary key or identifier columns that uniquely identify rows
    # * Always include date/time columns when present in the schema

    # - Generate comprehensive queries that provide sufficient context for downstream LLMs:
    # * When a user asks for a specific metric or attribute, include related fields that help contextualize that data
    # * Always include 2-4 additional relevant columns beyond what was explicitly requested
    # * For numerical data, include related metrics when they provide useful context
    # * For time-based queries, include sequential periods for trend analysis when possible

    # - Label your columns clearly in the SELECT statement to avoid ambiguity between multiple query results
    # - Use precise conditions in WHERE clauses to filter data based on the user's request
    # - Order results logically (chronologically for time series, alphabetically for categorical comparisons)
    # - Do not omit the main columns requested by the user

    # Database Schema:         
    # {column_str}          

    # Categorical Mappings:         
    # {json.dumps(categorical_values, indent=2)}          

    # User Request:         
    # {sub_prompt}     
    # """


    response = data_llm.invoke(llm_prompt).content.strip()
    return response


def execute_queries(sql_prompts):
    """
    Executes multiple SQL queries and merges results.
    """
    combined_df = pd.DataFrame()
    data_dict = {}

    for p in sql_prompts:
        prompt = p['prompt']
        table_names = p['tables']

        # Collect relevant tables into data_dict
        data_dict = {table: dataframes[table] for table in table_names if table in dataframes}

        # required_data = []
        # for table in table_names:
        #     required_data.append(dataframes[table])

        sql_query = generate_sql_query(prompt, data_dict)
        print(sql_query)
        
        result_df = retrive_releavant_data(sql_query,data_dict)
        
        if not result_df.empty:
            result_df["query_step"] = prompt  # Tag data with its step
            combined_df = pd.concat([combined_df, result_df], ignore_index=True)
            # combined_df.append(result_df)

    return combined_df



def summarize_data_result(result_df, user_prompt):
    """
    Summarizes the retrieved SQL data.
    """


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
    # for d in results_df:
    #     result_json = d.to_json()


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
    - Return the JSON in a format that is fully compatible with Python's `json.loads()`.
    - Do NOT include JavaScript functions inside the JSON (e.g., inside "tooltip" callbacks).
    - If necessary, return function-based values as **strings** instead.
    - Ensure all keys and values are properly formatted as valid JSON data types (strings, numbers, arrays, objects, booleans, or null).
    - Do NOT use undefined variables or symbols that are not valid in JSON.

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

    -Add title for chart
    -Add x label and y label for the charts in the chart.js format 
    

    Output: A valid, well-formatted, and easily readable JSON object representing a Chart.js configuration, ready to be used in a new Chart() constructor. The output must include properly configured axis labels based on the provided input. This includes adding titles and units to both axes within the options.scales section of the JSON.

    The response must:
    Follow JSON standard syntax.
    Be compatible with APIs like QuickChart.io.
    Not include any backticks or explicitly written JSON.

    ### User Request:
    {user_prompt}

    ### Data:
    {json.dumps(result_json)}

    """
    response = data_llm.invoke(prompt).content.strip()
    cleaned_response = re.sub(r"^```json\n|\n```$", "", response).strip()



    try:
        result = json.loads(cleaned_response)
        qc.config = result

        
    except:
        result = {}
        qc.config = result
        print("not possible")
    return qc.get_url(),result




def modify_visualisation(chart_json,user_prompt):
    """
    converts the current visualisation format to a more 
    """

    prompt = f"""   

    Your Job is to modify the given chart js format along with user prompt given to add modifications if required based on th euser input modify the current chart js data and reproduce new chart js data

    ### User Request:
    {user_prompt}

    ### Data:
    {chart_json}


    ### Instructions:
     Ensure the output is **only** the JSON  without any extra text. Do not include any backticks or json written just return the json 
     Follows JSON standard syntax, which is compatible with APIs like QuickChart.io.
     Uses double quotes (") for keys and strings, which is  valid JSON.
    - Return the JSON in a format that is fully compatible with Python's `json.loads()`.
    - Do NOT include JavaScript functions inside the JSON (e.g., inside "tooltip" callbacks).
    - If necessary, return function-based values as **strings** instead.
    - Ensure all keys and values are properly formatted as valid JSON data types (strings, numbers, arrays, objects, booleans, or null).
    - Do NOT use undefined variables or symbols that are not valid in JSON.

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
    

    -Add title for chart
    -Add x label and y label for the charts in the chart.js format 

    Output: A valid, well-formatted, and easily readable JSON object representing a Chart.js configuration, ready to be used in a new Chart() constructor. The output must include properly configured axis labels based on the provided input. This includes adding titles and units to both axes within the options.scales section of the JSON.
    


    The response must:
    Follow JSON standard syntax.
    Be compatible with APIs like QuickChart.io.
    Not include any backticks or explicitly written JSON.
    """

    res = gclient.chat.completions.create(
        model = 'llama3-70b-8192',
        messages=[   
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        response_format={'type':"json_object"},
    )

    


    result = json.loads(res.choices[0].message.content)

    qc.config = result
    return qc.get_url(),result







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
        visualisation,chart_json = visusalize(combined_df,visualisation_prompt)
        print(visualisation)
    else:
        visualisation = None
        chart_json = None


    return combined_df, summary ,visualisation,chart_json


if __name__ == "__main__":

    combined_df,summary,visualisation,chart_json = process_user_query("Show me the trend of EPS of google after 2021")

    print(combined_df.to_json())

    print("*******************************************************************************")

    print(summary)

    print("*********************************************************************************")

    print(visualisation)

    print("**********************************************************************************")

    print(chart_json)

    print("***********************************************************************************")

    link,output = modify_visualisation(chart_json,'The current represntation is a line chart resentation I would like to a have bar chart representation along with proper legends and x axis and y axis labels')

    print(link)


