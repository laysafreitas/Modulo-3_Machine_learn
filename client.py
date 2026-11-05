import requests

payload = {
"Unnamed: 0": 0,
    "id": 77777,
    "Age": 40,
    "Class": 2, 
    "Flight Distance": 3500, 
    "Inflight wifi service": 5,
    "Departure/Arrival time convenient": 5,
    "Ease of Online booking": 5,
    "Gate location": 5,
    "Food and drink": 5,
    "Online boarding": 5,
    "Seat comfort": 5,
    "Inflight entertainment": 5,
    "On-board service": 5,
    "Leg room service": 5,
    "Baggage handling": 5,
    "Checkin service": 5,
    "Inflight service": 5,
    "Cleanliness": 5,
    "Departure Delay in Minutes": 0,
    "Arrival Delay in Minutes": 0.0,
    "Gender_Male": True,
    "Customer Type_disloyal Customer": False,
    "Type of Travel_Personal Travel": False   
}

response = requests.post("http://127.0.0.1:8000/predict", json=payload)

print("Status Code:", response.status_code)
print("Resposta da API:", response.json())