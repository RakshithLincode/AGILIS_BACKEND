from pymodbus.client.sync import ModbusSerialClient as MC
import sys
import serial.tools.list_ports as list_ports

def list_all_ports():
	for port,pid,hwid in sorted(list_ports.comports()):
		print(port,pid,hwid)
	if not list_ports.comports():
		print('No devices found')

class modbus_ascii():
	def __init__(self,hwid):
		connection_status = self.initialise(hwid)
		if connection_status == False:
			print('Modbus Not connected')
			sys.exit(0)
		elif connection_status == True:
			print('Modbus connected')

	def initialise(self,hwid):
		for port,pid,hw_id in sorted(list_ports.comports()):
			try:
				#hw_id.split()
				#print(hw_id.split())
				#print((hw_id.split())[1])
				#print((hw_id.split())[1].split('='))
				if ((hw_id.split())[1].split('='))[1]:
					com = port
					print(port)
					break
			except:
				pass
		self.client=MC(method='ascii',port=com,timeout=1,parity='E',stopbits=1,bytesize=7,baudrate=9600)
		return self.client.connect()

	def read(self,address):
		return (self.client.read_coils(address, 1, unit=0x1)).bits[0]

	def write(self,address,val):
		self.client.write_coils(address, val, unit=0x1)

		return self.read(address) == val

		#2049 capture image
		#2059 start rotate
		#2060 reset after rejection
