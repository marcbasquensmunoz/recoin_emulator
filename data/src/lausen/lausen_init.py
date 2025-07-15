from data.src.lausen.credentials import *
from data.src.lausen.database_manager import DBClient
from enum import Enum

class Field(Enum):
    q_all = "PowerActualAll"
    q_all_calc = "PowerActualAllCalc"
    meanTb1 = "A1"
    meanTb2 = "A2"
    meanTb3 = "A3"
    HP_P = "HP-PowerActualHP"
    airT = "OutdoorTemp"
    dT1 = "FlowTempDivCalcA1"
    dT2 = "FlowTempDivCalcA2"
    dT3 = "FlowTempDivCalcA3"
    Tfin1 = "FlowTempForwardA1"
    Tfin2 = "FlowTempForwardA2"
    Tfin3 = "FlowTempForwardA3"
    Tfout1 = "FlowTempReturnA1"
    Tfout2 = "FlowTempReturnA2"
    Tfout3 = "FlowTempReturnA3"
    flowRate = "FlowRateActualAll"

    def is_averaged_magnitude(self):
        return self in {Field.meanTb1, Field.meanTb2, Field.meanTb3}

client = DBClient(url, token, org, bucket)
