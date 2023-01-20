import subprocess
from subprocess import PIPE

proc = subprocess.run(['docker','ps','-aq'],check=True,stdout=PIPE,encoding='ascii')
container_ids = proc.stdout.strip().split()
if container_ids:
    subprocess.run(['docker','stop']+container_ids,check=True)
    subprocess.run(['docker','rm']+container_ids,check=True)
    
