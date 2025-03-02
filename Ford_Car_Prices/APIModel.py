from fastapi import FastAPI
import joblib
from pydantic import BaseModel, Field
import numpy as np
import uvicorn
import pandas as pd

model = joblib.load("regression_model.pkl")

app = FastAPI()

class CementModel(BaseModel):
    Cement: int
    Blast_Furnace_Slag: int = Field(..., alias="Blast Furnace Slag")
    Fly_Ash: int = Field(default=0, alias="Fly Ash")
    Water: int
    Superplasticizer: int
    Coarse_Aggregate: int = Field(..., alias="Coarse Aggregate")
    Fine_Aggregate: int = Field(..., alias="Fine Aggregate")
    Age: int = Field(..., alias="Age (day)")

@app.post('/predict')
def predict_response(data: CementModel):
    
    features = pd.DataFrame([{
        "Cement": data.Cement,
        "Blast Furnace Slag": data.Blast_Furnace_Slag,
        "Fly Ash": data.Fly_Ash,
        "Water": data.Water,
        "Superplasticizer": data.Superplasticizer,
        "Coarse Aggregate": data.Coarse_Aggregate,
        "Fine Aggregate": data.Fine_Aggregate,
        "Age (day)": data.Age
    }])

    
    prediction = model.predict(features)

    return {"predicted_strength": prediction[0]}


if __name__ == "__main__":
    uvicorn.run(app)

