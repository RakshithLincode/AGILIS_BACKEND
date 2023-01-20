import os

sudoPassword = '123456789'
command_1 = '/home/schneider/anaconda3/envs/livis/bin/python /home/schneider/deployment25Nov/livis/plc_service/plc_service.py'
p = os.system('echo %s|sudo -S %s' % (sudoPassword, command_1))


#echo 123 | sudo -S python3 /home/administrator/Documents/printy.py
