import os
from data.src.lausen.lausen_init import client

borehole = "A1"
timeStart = "2025-01-01T00:00:00Z"
timeStop = "2025-01-31T00:00:00Z"
df = client.get_temperature_profile_data(borehole, timeStart, timeStop, "1h")
df.to_csv(f"{os.getcwd()}/data/data/temperature_profile_{borehole}.csv", index=False)
