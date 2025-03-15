import pandas as pd
from pandasql import sqldf

# Load the dataset
data = pd.read_csv(r'data.csv')

# 1️⃣ Function to return column names and their data types
# def get_column_info():
#     df = data.copy()
#     return df.dtypes.to_dict()


def get_column_info():
    """
    Extracts column names and data types dynamically.
    Also retrieves unique values for categorical columns (e.g., company names).
    """
    df = data.copy()
    column_info = df.dtypes.to_dict()

    # Collect possible categorical values (e.g., company names, categories, etc.)
    categorical_values = {}
    for col in df.select_dtypes(include=["object"]).columns:
        unique_vals = df[col].dropna().unique().tolist()
        categorical_values[col] = unique_vals[:50]  # Limit for efficiency

    return column_info, categorical_values



# 2️⃣ Function to retrieve data based on JSON query
def filter_data(df, query_conditions):
    """
    query_conditions: Dictionary where keys are column names and values are filtering conditions.
    Example:
        {
            "Customer ID": "CUST1234",
            "Order Type": "Online"
        }
    """
    filtered_df = df.copy()
    for column, value in query_conditions.items():
        if column in df.columns:
            filtered_df = filtered_df[filtered_df[column] == value]
    return filtered_df



def retrive_releavant_data(data_query):
    df = data.copy()
    filtered_df = sqldf(data_query, {"data":df})
    return filtered_df


# query = """
# SELECT `Earnings per Share` FROM data WHERE `Calendar Year` BETWEEN 2019 AND 2023 AND `Ticker` = 'GOOG'

# """
# print(retrive_releavant_data(query))
# print(get_column_info())

# print(data.describe(include = 'all'))

# print(data['Ticker'].unique())

# print(get_column_info())