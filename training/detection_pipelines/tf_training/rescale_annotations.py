import cv2
import glob
import os
import glob
import xml.etree.ElementTree as ET

def rescale_annotations(dataset_path, scale):
	for i in glob.glob('{}/*/*.jpg'.format(dataset_path)):

		######### Editing images #########
		img = cv2.imread('{}'.format(i))
		img = cv2.resize(img,(0,0),fx=scale,fy=scale)
		cv2.imwrite(str(i),img)

		####### Editing XML file ##########
		xml_file = i.replace('.jpg','.xml')
		tree = ET.parse(xml_file)

		tree.find('.//width').text = str(int(int(tree.find('.//width').text)*scale))
		tree.find('.//height').text = str(int(int(tree.find('.//width').text)*scale))

		for i in (tree.iterfind('.//xmin')): 
			i.text = str(int(int(i.text)*scale))
		
		for i in (tree.iterfind('.//xmax')): 
			i.text = str(int(int(i.text)*scale))
		
		for i in (tree.iterfind('.//ymin')): 
			i.text = str(int(int(i.text)*scale))
		
		for i in (tree.iterfind('.//ymax')): 
			i.text = str(int(int(i.text)*scale))

		tree.write(xml_file)
	print('Done rescaling annotations')

dataset_path = "/home/aniruddh/lincode/UBSpirit/ttt"
scale = 0.7
rescale_annotations(dataset_path, scale)