import os
from data.src.lausen.lausen_init import client, Field

timeStart = "2024-11-05T00:00:00Z"
timeStop = "2025-05-31T00:00:00Z"
df = client.get_field_data([Field.flowRate, Field.q_all, Field.meanTb1, Field.Tfin1], timeStart, timeStop)
df.to_csv(f"{os.getcwd()}/data/data/internal_model_data.csv", index=False)
