import neoapi
import datetime
import cv2
from arena_api.system import system
from arena_api.buffer import *
from arena_api import enums
import time
import numpy as np

class camera():
	def __init__(self,type,id):
		if type == 'baumer':
			self.camera = baumer(id) 
		elif type == 'webcam':
			pass
	def fetch_image(self):
		return self.camera.fetch_image()

class baumer():
	def __init__(self,ip):
		self.timeout = 1
		self.camera = neoapi.Cam()
		self.camera.Connect(ip)
		self.camera.f.PixelFormat.SetString('BayerRG8')
		self.camera.f.TriggerMode = neoapi.TriggerMode_On
		self.camera.f.TriggerSource = neoapi.TriggerSource_Software

	def fetch_image(self):
		self.camera.f.TriggerSoftware.Execute()
		start = datetime.datetime.now()
		while True:
			if datetime.datetime.now() - start >= datetime.timedelta(seconds= self.timeout):
				return 'timeout'
			img = self.camera.GetImage().GetNPArray()
			if len(img)==0:
				continue
			else:
				break
		print(datetime.datetime.now() - start)
		img = cv2.cvtColor(img, cv2.COLOR_BAYER_RG2RGB)
		return img

class Lucid():	
	def get_ip(self,device1):
		self.raw_ip = device1.nodemap['GevCurrentIPAddress'].value
		self.ip_list = []
		for i in range(4):
			self.temp = str(int(self.raw_ip/(256**(3-i))%256))
			self.ip_list.append(self.temp)
		self.final_ip = self.ip_list[0]+'.'+self.ip_list[1]+'.'+self.ip_list[2]+'.'+self.ip_list[3]
		return self.final_ip

	def get_ser(self,device1):
		self.ser = device1.nodemap['DeviceSerialNumber'].value
		print(self.ser)
		return self.ser

	def __init__(self,ser,pos):
		self.devices = system.create_device()
		self.camera_ips = {ser:pos}
		self.cameras_dict = {}
		for d in self.devices:
			temp = self.get_ser(d)
			self.cameras_dict[self.camera_ips[temp]] = d	

	tries = 0
	tries_max = 6
	sleep_time_secs = 10
	while tries < tries_max:  # Wait for device for 60 seconds
		devices = system.create_device()
		if not devices:
			print(
				f'Try {tries+1} of {tries_max}: waiting for {sleep_time_secs} '
				f'secs for a device to be connected!')
			for sec_count in range(sleep_time_secs):
				time.sleep(1)
				print(f'{sec_count + 1 } seconds passed ',
					'.' * sec_count, end='\r')
			tries += 1
		else:
			print(f'Created {len(devices)} device(s)\n')
			device = devices[0]
			break
	else:
		raise Exception(f'No device found! Please connect a device and run '
						f'the example again.')
    
	def fetch_cameras(self):
		self.initial_acquisition_mode_list = []
		for device in self.devices:
			self.nodemap = device.nodemap
			self.initial_acquisition_mode_list.append(self.nodemap.get_node("AcquisitionMode").value)
			self.nodemap.get_node("AcquisitionMode").value = "Continuous"
			self.tl_stream_nodemap = device.tl_stream_nodemap
			self.tl_stream_nodemap["StreamBufferHandlingMode"].value = "NewestOnly"
			self.tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
			self.tl_stream_nodemap['StreamPacketResendEnable'].value = True
			self.nodes = self.nodemap.get_node(['PixelFormat'])
			self.nodes['PixelFormat'].value = 'RGB8'
		while 1:
			for cam_pos,device in self.cameras_dict.items():
				with device.start_stream(1):
					self.buffer = device.get_buffer()
					self.item = BufferFactory.copy(self.buffer)
					device.requeue_buffer(self.buffer)
					self.npndarray = np.ctypeslib.as_array(self.item.pdata,shape=(self.item.height, self.item.width, 3)).reshape(self.item.height, self.item.width, 3)
					self.npndarray = cv2.cvtColor(self.npndarray, cv2.COLOR_BGR2RGB)
					self.frame = self.npndarray
					# print(self.frame)
					return self.frame

	    						


