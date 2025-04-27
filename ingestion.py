import pandas as pd
from pandasql import sqldf
import polars as pl
import os
# Load the dataset
# data = pd.read_csv(r'C:\Desktop\experiments\data_analytics\database\Financial_corporate_data.csv')

data = pd.read_csv(r'C:\Desktop\experiments\data_analytics\database\sales_data.csv',encoding="ISO-8859-1")
print(data.columns)

# 1️⃣ Function to return column names and their data types


def get_dataframes(folder_path =r"C:\Desktop\experiments\data_analytics\database" ):
    csv_dataframes = {}

    # Iterate through files in the folder
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):  # Check if the file is a CSV
            file_path = os.path.join(folder_path, file)
            try:
            
                df = pd.read_csv(file_path)  # Read CSV file into a dataframe
            except:
                df = pd.read_csv(file_path,encoding="ISO-8859-1")
            csv_dataframes[os.path.splitext(file)[0]] = df  # Store in dictionary

    return csv_dataframes





def get_column_info(data_dict):
    """
    Extracts column data types and categorical mappings for all tables in data_dict.
    """
    column_info = {}
    categorical_values = {}

    for table_name, df in data_dict.items():
        column_info[table_name] = {col: str(df[col].dtype) for col in df.columns}

        # Extract categorical values for non-numeric columns (if needed)
        # categorical_values[table_name] = {
        #     col: df[col].unique().tolist()
        #     for col in df.select_dtypes(exclude=['number']).columns
        # }
        try:
            categorical_values[table_name] = {
                col: df[col].unique()[:30].tolist()
                # for col in df.columns if df[col].dtype == pl.Utf8
                for col in df.select_dtypes(exclude=['number']).columns
            }

        #scenariosn where tehre are less than 30 categorical values 
        except:
             categorical_values[table_name] = {
                col: df[col].unique().tolist()
                # for col in df.columns if df[col].dtype == pl.Utf8
                for col in df.select_dtypes(exclude=['number']).columns
            }



    return column_info, categorical_values


def get_columns(file_path):
    # df = pd.read_csv(file_path, encoding="ISO-8859-1")

    try:
        df = pd.read_csv(file_path)
    except:
        df = pd.read_csv(file_path, encoding="ISO-8859-1")

    
    # Convert dtypes to string so they are JSON serializable
    column_info = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    return column_info


# def get_columns(file_path):
#     try:
#         df = pl.read_csv(file_path)
#     except:
#         df = pl.read_csv(file_path, encoding="ISO-8859-1")

#     column_info = {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}
#     return column_info




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



def retrive_releavant_data(data_query,dicto={"data":data}):
    # df = data.copy()
    filtered_df = sqldf(data_query, dicto)
    return filtered_df

# def retrive_releavant_data(data_query,required_data):
#     filtered_df = required_data[0].sql(data_query).to_pandas()
#     return filtered_df


query = """
SELECT `Earnings per Share` FROM self WHERE `Calendar Year` BETWEEN 2019 AND 2023 AND `Ticker` = 'GOOG'

"""

# query = """
# SELECT 
#   ORDERNUMBER AS OrderNumber,
#   STATUS AS Status,
#   ORDERDATE AS OrderDate,
#   QTR_ID AS QuarterID,
#   YEAR_ID AS YearID,
#   COUNT(ORDERNUMBER) AS ShippedOrdersCount
# FROM
#   self

# """



# print(data.sql(query).to_pandas())

# print(retrive_releavant_data(query,{"data":data}))
# a,b = get_column_info({"data":data})

# print(b)

# print(data.describe(include = 'all'))

# print(data['Ticker'].unique())

# print(get_column_info())

# print(data.select_dtypes(include = ['category']))
