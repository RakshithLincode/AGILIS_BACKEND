import serial
import time
import json

ser = serial.Serial("/dev/ttyACM0")

while 1:
	if json.load(open('tower_status.json'))==1:
		ser.write(b'1')
		
		print(1)
	elif json.load(open('tower_status.json'))==2:
		ser.write(b'2')
		print(2)
		
	elif json.load(open('tower_status.json'))==0:
		ser.write(b'0')
		ser.close() 
		
	time.sleep(1)