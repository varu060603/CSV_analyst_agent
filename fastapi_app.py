from app import process_user_query,modify_visualisation
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Any,List
import os


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class UserPrompt(BaseModel):
    prompt: str

class modify_vis(BaseModel):
    user_prompt:str
    chart_json: Any


@app.post("/display")
async def display(folder_path =f"C:\Desktop\experiments\data_analytics\database" ):
    """
    Returns a list of all CSV file names in the given folder.
    """
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    return csv_files



@app.post("/analyze")
async def analyze(user_prompt: UserPrompt):
    combined_df, summary, visualisation, chart_json = process_user_query(user_prompt.prompt)
    combined_df = combined_df.to_json()
    return {
        "summary": summary,
        "chart_json": chart_json,
        "visualisation_link": visualisation,
        "table": combined_df
    }



@app.post("/modify_visualisation")
async def modify_visualisation_endpoint(input:modify_vis):
    visualisation_link,chart_json = modify_visualisation(input.chart_json,input.user_prompt)
    return {
        "visualisation_link":visualisation_link,
        "chart_json":chart_json
    }




if __name__ == "__main__":
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
