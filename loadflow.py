import pypsa
import numpy as np
from sklearn.preprocessing import MinMaxScaler

##### REFERENCE WEBSITE: https://pypsa.org/doc/quick_start.html #########

network=pypsa.Network()

for i in range(5):
	network.add("Bus","mybus{}".format(i+1))

print network.buses

network.add("Line","myline1",bus0="bus1",bus1="bus2",r=0.02,x=0.06)
network.add("Line","myline2",bus0="bus1",bus1="bus3",r=0.08,x=0.24)
network.add("Line","myline3",bus0="bus2",bus1="bus3",r=0.06,x=0.25)
network.add("Line","myline4",bus0="bus2",bus1="bus4",r=0.06,x=0.18)
network.add("Line","myline5",bus0="bus2",bus1="bus5",r=0.04,x=0.12)
network.add("Line","myline6",bus0="bus3",bus1="bus4",r=0.01,x=0.03)
network.add("Line","myline7",bus0="bus4",bus1="bus5",r=0.08,x=0.24)

print network.lines

network.add("Generator","mygen",bus="bus2",p_set=#########)
print network.generators
print network.generators.p_set

network.add("Load","myload1",bus="mybus2",p_set=######)

print network.loads

print network.loads.p_set

#do a Newton Raphson power flow
network.pf()

#print the flow values for each line
print network.lines_t.p0

# run this over "hourly" iterations with the values from every bus
# compare with predicted model from the powerpredict.py program
