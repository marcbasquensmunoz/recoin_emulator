import influxdb_client
import pandas as pd

class DBClient():
    def __init__(self, url, token, org, bucket):
        self.client = influxdb_client.InfluxDBClient(
            url=url,
            token=token,
            org=org
        )
        self.bucket = bucket
        self.org = org
        self.measurement = "SilanaData"

    def parse_data(self, tables):
        df = pd.DataFrame([
            {
                **record.values
            }
            for table in tables
            for record in table.records
        ])
        df = df.drop(columns=["result", "table"])
        return df

    def execute_query(self, query):
        query_api = self.client.query_api()
        result = query_api.query(org=self.org, query=query)
        return self.parse_data(result)
    
    def build_base_query(self, timeStart, timeStop):
        return f'''from(bucket: "{self.bucket}")
        |> range(start: {timeStart}, stop: {timeStop})
        |> filter(fn: (r) => r["_measurement"] == "{self.measurement}")'''
    
    def filter_device(self, device):
        return f'|> filter(fn: (r) => r["Device"] == "{device}")'

    def filter_borehole(self, borehole):
        return f'|> filter(fn: (r) => r["BoreHole"] == "{borehole}")'

    def filter_channel(self, channel):
        return f'|> filter(fn: (r) => r["ChName"] == "{channel}")'
    
    def get_field_query(self, field, timeStart, timeStop, dt):
        if field.is_averaged_magnitude():
            field_query = f"""{self.filter_borehole(field.value)}
                              {self.filter_device("Laser")}"""
        else:
            field_query = self.filter_channel(field.value)

        return f"""{self.build_base_query(timeStart, timeStop)}
        {field_query}
        |> map(fn: (r) => ({{
            _time: r["_time"],
            _value: r["_value"],
            _field: r["ChName"]
        }}))
        |> keep(columns: ["_value", "_time"])
        |> aggregateWindow(every: {dt}, fn: mean, createEmpty: false)
        """
    
    def get_profile_query(self, borehole, timeStart, timeStop, dt):
        return f"""{self.build_base_query(timeStart, timeStop)}
            {self.filter_borehole(borehole)}
            {self.filter_device("Laser")}
            |> aggregateWindow(every: {dt}, fn: mean, createEmpty: false)
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            |> keep(columns: ["_time", "BoreHole", "ChName", "Deep", "DeepCounter", "Direction", "ChValue"])
        """

    def parse_dt_for_pd(self, dt: str):
        if 'm' in dt:
            return dt.replace("m", "min")
        return dt

    def get_field_data(self, fields, timeStart, timeStop, dt="20m"):
        df = pd.DataFrame()    
        expected_timestamps = pd.date_range(start=timeStart, end=timeStop, freq=self.parse_dt_for_pd(dt), tz='UTC')[1:]
        df["_time"] = expected_timestamps
        df = df.reset_index(drop=True)
        df.index.name = None

        for field in fields:
            query = self.get_field_query(field, timeStart, timeStop, dt)
            print(query)
            query_result = self.execute_query(query)
            col_name = field.name
            query_result = query_result.rename(columns={"_value": col_name})
            df = pd.merge(left=df, right=query_result[["_time", col_name]], how='left', on="_time")

        return df
    
    def get_temperature_profile_data(self, borehole, timeStart, timeStop, dt="20m"):
        query = self.get_profile_query(borehole, timeStart, timeStop, dt)
        return self.execute_query(query)
