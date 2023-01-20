sudo docker rm -f monk
sudo docker run --name monk -d -t trial 
sudo docker exec -ti monk bash -c "/root/anaconda3/envs/monk/bin/python3.6 /home/monk/Monk_Object_Detection/13_tf_obj_2/lib/a.py"
sudo service docker restart
