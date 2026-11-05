from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np

try:
    model = joblib.load("modelo_occupancy_rf.joblib")
except Exception as e:
    raise RuntimeError(f"Erro ao carregar o modelo: {e}")

app = FastAPI(
    title="API de Satisfação de Passageiros",
    version="2.4"
)

@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return Response(status_code=204)

class PassengerData(BaseModel):
    unnamed_0: int = Field(..., alias="Unnamed: 0")
    id: int
    Age: int
    Class: int
    flight_distance: int = Field(..., alias="Flight Distance")
    inflight_wifi_service: int = Field(..., alias="Inflight wifi service")
    departure_arrival_time_convenient: int = Field(..., alias="Departure/Arrival time convenient")
    ease_of_online_booking: int = Field(..., alias="Ease of Online booking")
    gate_location: int = Field(..., alias="Gate location")
    food_and_drink: int = Field(..., alias="Food and drink")
    online_boarding: int = Field(..., alias="Online boarding")
    seat_comfort: int = Field(..., alias="Seat comfort")
    inflight_entertainment: int = Field(..., alias="Inflight entertainment")
    on_board_service: int = Field(..., alias="On-board service")
    leg_room_service: int = Field(..., alias="Leg room service")
    baggage_handling: int = Field(..., alias="Baggage handling")
    checkin_service: int = Field(..., alias="Checkin service")
    inflight_service: int = Field(..., alias="Inflight service")
    Cleanliness: int
    departure_delay_in_minutes: int = Field(0, alias="Departure Delay in Minutes")  
    arrival_delay_in_minutes: float = Field(0.0, alias="Arrival Delay in Minutes")   
    Gender_Male: bool
    customer_type_disloyal_customer: bool = Field(..., alias="Customer Type_disloyal Customer")
    type_of_travel_personal_travel: bool = Field(..., alias="Type of Travel_Personal Travel")

    class Config:
        populate_by_name = True

@app.get("/")
def home():
    return {"message": "API rodando com todas as features exigidas!"}

@app.post("/predict")
def predict(data: PassengerData):
    try:
        input_data = pd.DataFrame([data.dict(by_alias=True)])
        
        bool_cols = ['Gender_Male', 'Customer Type_disloyal Customer', 'Type of Travel_Personal Travel']
        for col in bool_cols:
            if col in input_data.columns:
                input_data[col] = input_data[col].astype(int)

        if hasattr(model, "feature_names_in_"):
            input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)
        
        class_id = int(prediction[0])
        class_name = "satisfeito" if class_id == 1 else "neutro ou insatisfeito"
        
        return {
            "prediction_code": class_id,
            "prediction_label": class_name,
            "confidence": float(np.max(probability))
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))