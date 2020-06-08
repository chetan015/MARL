# cse_571_team8

First we need to train the model by updating the Q tables, then we can run the simulation. 
Every command below this point should be run in a separate terminal. When you train, the Q table of each tbot will be saved, as well as the cumulative reward, like in homework #3.

First run each of these commands to download the code and give permissions. If you are not downloading from Github then skip the following line.

cd ~/catkin_ws/src && git clone https://github.com/AdamIshay/cse571_project.git

chmod u+x ~/catkin_ws/src/cse571_project/scripts/*.py

chmod u+x ~/catkin_ws/src/cse571_project/env_setup.sh && ~/catkin_ws/src/cse571_project/env_setup.sh



roscore



if running with 1 book for each tbot:

	
	if training:

		rosrun cse571_project server.py -sub 1 -b 1 -s 32 -t 1

		rosrun cse571_project qlearning.py -task 3 -episodes 450

	if running simulation:

		rosrun cse571_project server.py -sub 1 -b 1 -s 32 -t 0

		rosrun cse571_project move1_bot3.py --> for first bot

		rosrun cse571_project move2_bot3.py --> for second bot
		
		roslaunch cse571_project maze.launch

		rosrun cse571_project qlearning.py -task 2 -episodes 0

if running with 3 books for each tbot:
	
	rosrun cse571_project server.py -sub 1 -b 3 -s 32 -t 1

	if training:

		rosrun cse571_project server.py -sub 1 -b 3 -s 32 -t 1
	
		rosrun cse571_project qlearning.py -task 4 -episodes 450

	if running simulation:

		rosrun cse571_project server.py -sub 1 -b 3 -s 32 -t 0

		rosrun cse571_project move1_bot3.py --> for first bot

		rosrun cse571_project move2_bot3.py --> for second bot
		
		roslaunch cse571_project maze.launch

		rosrun cse571_project qlearning.py -task 5 -episodes 0



rosrun cse571_project move1_bot3.py --> for first bot

rosrun cse571_project move2_bot3.py --> for second bot

roslaunch cse571_project maze.launch

rosrun cse571_project qlearning.py -task 1 --> As of now, we tested in this.
