from modbus_ascii import modbus_ascii
# pb = 2048
# reset = 2058
# y4_on = 2052
# y4_off = 2053
# y4_status = 2054
# y4 = [y4_on,y4_off,y4_status]
# y5_on = 2055
# y5_off = 2056
# y5_status = 2057
# y5 = [y5_on,y5_off,y5_status]
# y=[0,0,0,y3,y4,y5]



class plc_io():
	def __init__(self):
		self.m=modbus_ascii('1A86:7523')

	def read(self,address):
		return self.m.read(address)

	def write(self,address,val):
		status = self.m.write(address,val)
		return status

		


