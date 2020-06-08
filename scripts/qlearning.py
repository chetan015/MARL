#!/usr/bin/env python
# encoding: utf-8

import rospy
from std_msgs.msg import String
import problem
import json
import os
import argparse
import numpy as np
import random
import environment_api as api
from matplotlib import pyplot as plt
import pdb
import re
from Agent import Agent

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-task', help="Task to execute:\n1. Q learning on sample trajectories\n2. Q learning without pruned actions\n3. Q learning with pruned actions", metavar='1', action='store', dest='task', default="1", type=int)
parser.add_argument("-sample", metavar="1", dest='sample', default='1', help="which trajectory to evaluate (with task 1)", type=int)
parser.add_argument('-episodes', help="Number of episodes to run (with task 2 & 3)", metavar='1', action='store', dest='episodes', default="1", type=int)
parser.add_argument('-headless', help='1 when running in the headless mode, 0 when running with gazebo', metavar='1', action='store', dest='headless', default=1, type=int)


class QLearning:

    def __init__(self, task, headless=1, sample=1, episodes=1):
        rospy.init_node('qlearning', anonymous=True)
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
        
        self.books_json_file = root_path + "/books.json"
        self.books = json.load(open(self.books_json_file))
        self.helper = problem.Helper()
        self.helper.reset_world()

        self.headless = headless
        self.alpha = 0.3
        self.gamma = 0.9
        self.root_path = root_path

        if(task == 1):
            trajectories_json_file = root_path + "/trajectories{}.json".format(sample)
            q_values = self.task1(trajectories_json_file)
        elif(task == 2):
            q_values = self.task2(episodes)
        elif(task == 3):
            q_values = self.task3(episodes)
        elif(task==4):
            q_values = self.task4(episodes)
        elif(task==5):
            q_values=self.task5(episodes)

        with open(root_path + "/q_values.json", "w") as fout:
            json.dump(q_values, fout)

    def reward_prune(self,current_state,state_tbot,state_tbot_other,action,action_items,tbot,tbot_other):
        #function that chooses to prune invalid actions (with large negative reward and no state change) or go through the api 
        through_api=True
        next_state=None
        reward=None
        
        if action!='moveF' and action!='pick':
            return through_api,next_state,reward
        
        if action=='pick':
            book_number=int(re.findall(r'\d+', action_items[0])[0])
            if book_number not in tbot.books:
                reward=self.book_penalty
                through_api=False
                next_state=current_state.copy()

        if action=='moveF':
            tbot_near=tbot.tbot_near(state_tbot,state_tbot_other)
            if tbot_near==state_tbot[2]: #if another tbot is in the same direction that tbot_active is facing
                reward=self.bump_penalty
                through_api=False
                next_state=current_state.copy()
            

        return through_api,next_state,reward
    
    def choose_action(self, tbot,epsilon,state):
        if epsilon>np.random.uniform(): #if this, do random
            action_string=random.choice(api.get_all_actions())
            
        else:
            #pdb.set_trace()
            action_idx=tbot.q[state].argmax()
            action_string=tbot.idx_to_action[action_idx]
            
        action_split=action_string.split()
        action=action_split[0]
        action_items=action_split[1:]
        
        action_params={}
        for i,item in enumerate(self.action_reference[action]['params']):
            action_params[str(item)]=action_items[i]
        
        return action,action_items,action_params,action_string
    
    
    def task2(self, episodes):
        
        q_values = {}
        
        
        actions_json_file='/action_config.json'

        with open(self.root_path + actions_json_file) as json_file:
            try:
                self.action_reference = json.load(json_file, parse_float=float)
            except (ValueError, KeyError, TypeError):
                print "JSON error"
        
        
        
# =============================================================================
#       episode parameters
# =============================================================================
        #there are actually 2*episodes episodes, since there are two tbots
        episode_update=2 #the amount of episodes a tbot will train while the other tbots policy remains constant. must be a divisor of episodes
        episode_blocks=int(episodes/episode_update) #number of episode blocks, each episode block one tbot is updating their q table while the other only acts 
# =============================================================================
#       epsilon parameters & set penalty values 
# =============================================================================
        epsilon_initial=.95
        epsilon_decay=.002
        epsilon_calc= lambda epsilon_initial,epsilon_decay,i: max(0.05, epsilon_initial - epsilon_decay*i) 
        
        
        self.book_penalty=-100
        self.bump_penalty=-100
        
        
# =============================================================================
#         q tables initialized to zero 
# =============================================================================
        q1=np.zeros((7,7,4,2,2,5,5)) #(x,y, orientation, c1,c2,tbot_near,action) #c1,c2 will be zero if available, and one if picked up 
        q2=np.zeros((7,7,4,2,2,5,5))
        
# =============================================================================
#       Create Agents
# =============================================================================
        
        agent1_books=[1]
        agent2_books=[2]
        
        agent1=Agent('robot1',q1,agent1_books) 
        agent2=Agent('robot2',q2,agent2_books)
        
        tbot_list=[agent1,agent2]
        
        
        R_cumulative={agent1.name:[],agent2.name:[]}
        
# =============================================================================
#       acting and training 
# =============================================================================
        
        for i in range(episode_blocks):
          epsilon=epsilon_calc(epsilon_initial,epsilon_decay,i) #epsilon can only changes every episode block
          for tbot in tbot_list: #determines which tbot is learning, active updates table, passive does not
            tbot_active=tbot
            tbot_passive_set=set(tbot_list)-set([tbot])
            tbot_passive=tbot_passive_set.pop()
            for e in range(episode_update):#cycle through the episodes inside an episode block
              epsilon=epsilon_calc(epsilon_initial,epsilon_decay,i*episode_update+e)
              # a single episode
              api.reset_world()
              R_cumulative_active=0
              R_cumulative_passive=0
              initial_state=api.get_current_state()
              current_state=initial_state
              
              state_active=tbot_active.dict_to_np_state(current_state,tbot_passive)  #active bots state tuple
              state_passive=tbot_passive.dict_to_np_state(current_state,tbot_active) #passive bots state tuple
              #pdb.set_trace()
              print('episode_block {0} episode {1} for tbot {2}'.format(i,e, tbot_active.name))
              while not api.is_terminal_state(current_state):
                
                through_api=True # flag for going through API for an action, if False, then reward is given manually

# =============================================================================
#               active tbot acts and learns 
# =============================================================================
                #pick action for tbot_active
                #choose either random or exploit, according to epsilon=epsilon_calc(epsilon_initial, epsilon_decay, i)
# =============================================================================
#                 if state_active[0]>=5 or state_passive[0]>=5:
#                     pdb.set_trace()
# =============================================================================
                action_A,action_items_A,action_params_A,action_string_A=self.choose_action(tbot_active,epsilon,state_active) #selects action

                through_api,next_state,reward=self.reward_prune(current_state,state_active,state_passive,action_A,action_items_A,tbot_active,tbot_passive) #prunes by checking for invalid actions, in which case we don't run through environment_api 
                #pdb.set_trace()
                if through_api:
                  success,next_state=api.execute_action(action_A,action_params_A,tbot_active.name)
                  reward=api.get_reward(current_state,action_A,next_state)
                
                R_cumulative_active+=reward
                
                
                next_state_active=tbot_active.dict_to_np_state(next_state,tbot_passive)
                
                
                #update q_values of tbot_active ONLY
                
                state_action_idx=tuple(state_active)+tuple([tbot_active.idx_to_action.index(action_string_A)])
                tbot_active.q[state_action_idx]=(1-self.alpha)*tbot_active.q[state_action_idx]+self.alpha*(reward+self.gamma*max(tbot_active.q[next_state_active]))

                current_state=next_state # udpate current state so the other tbot knows the updated state before choosing an action etc.
                
                state_active=tbot_active.dict_to_np_state(current_state,tbot_passive)  #active bots state tuple
                state_passive=tbot_passive.dict_to_np_state(current_state,tbot_active) #passive bots state tuple
        
        
# =============================================================================
#               passive tbot acts and does NOT learn
# =============================================================================
                action_P,action_items_P,action_params_P,action_string_P=self.choose_action(tbot_passive,epsilon,state_passive)
        
                through_api,next_state,reward=self.reward_prune(current_state,state_passive,state_active,action_P,action_items_P,tbot_passive,tbot_active) #reward won't be used 
                
                if through_api:
                  success,next_state=api.execute_action(action_P,action_params_P,tbot_passive.name)
                  reward=api.get_reward(current_state,action_P,next_state)
        
                R_cumulative_passive+=reward
                current_state=next_state # udpate current state for active tbot
                
                state_active=tbot_active.dict_to_np_state(current_state,tbot_passive)  #active bots state tuple
                state_passive=tbot_passive.dict_to_np_state(current_state,tbot_active) #passive bots state tuple
        
              R_cumulative[tbot_active.name].append(R_cumulative_active)
              R_cumulative[tbot_passive.name].append(R_cumulative_passive)
              print(R_cumulative)
              print('on the {0}th episode is {1}'.format(i*episode_update+e, epsilon))
        np.save('q_table_r1.npy',agent1.q)
        np.save('q_table_r2.npy', agent2.q)
        
        import pickle
        with open("robot1_rewards.txt", "wb") as f:
            pickle.dump(R_cumulative['robot1'], f)
            
        with open("robot2_rewards.txt", "wb") as f:
            pickle.dump(R_cumulative['robot2'], f)
        
        return q_values

    def task3(self, episodes):
        '''for running the simulation after training'''
        q_values = {}
        # Your code here
        
        actions_json_file='/action_config.json'

        with open(self.root_path + actions_json_file) as json_file:
            try:
                self.action_reference = json.load(json_file, parse_float=float)
            except (ValueError, KeyError, TypeError):
                print "JSON error"
        
        self.book_penalty=-100
        self.bump_penalty=-100
        
        
# =============================================================================
#         q tables initialized to zero 
# =============================================================================
        #(x,y, orientation, c1,c2,tbot_near,action) #c1,c2 will be zero if available, and one if picked up 

        
        q1 = np.load('q_table_r1.npy')
        q2 = np.load('q_table_r2.npy')
# =============================================================================
#       Create Agents
# =============================================================================
        
        agent1_books=[1]
        agent2_books=[2]
        
        agent1=Agent('robot1',q1,agent1_books) 
        agent2=Agent('robot2',q2,agent2_books)
        
        tbot_list=[agent1,agent2]

        R_cumulative={agent1.name:[],agent2.name:[]}
        
# =============================================================================
#       acting 
# =============================================================================
        
        
      
        tbot_active=agent1
        tbot_passive=agent2
        epsilon=0
        # a single episode
        api.reset_world()
        R_cumulative_active=0
        R_cumulative_passive=0
        initial_state=api.get_current_state()
        current_state=initial_state
      
        state_active=tbot_active.dict_to_np_state(current_state,tbot_passive)  #active bots state tuple
        state_passive=tbot_passive.dict_to_np_state(current_state,tbot_active) #passive bots state tuple
        #pdb.set_trace()
      
        while not api.is_terminal_state(current_state):
        
            through_api=True # flag for going through API for an action, if False, then reward is given manually

# =============================================================================
#               active tbot acts and learns 
# =============================================================================
        #pick action for tbot_active
        #choose either random or exploit, according to epsilon=epsilon_calc(epsilon_initial, epsilon_decay, i)
# =============================================================================
#                 if state_active[0]>=5 or state_passive[0]>=5:
#                     pdb.set_trace()
# =============================================================================
            action_A,action_items_A,action_params_A,action_string_A=self.choose_action(tbot_active,epsilon,state_active) #selects action

            through_api,next_state,reward=self.reward_prune(current_state,state_active,state_passive,action_A,action_items_A,tbot_active,tbot_passive) #prunes by checking for invalid actions, in which case we don't run through environment_api 
            #pdb.set_trace()
            if through_api:
                success,next_state=api.execute_action(action_A,action_params_A,tbot_active.name)
                reward=api.get_reward(current_state,action_A,next_state)
        
            R_cumulative_active+=reward
        
        
            next_state_active=tbot_active.dict_to_np_state(next_state,tbot_passive)
        
        
            #update q_values of tbot_active ONLY
        
            state_action_idx=tuple(state_active)+tuple([tbot_active.idx_to_action.index(action_string_A)])

            current_state=next_state # udpate current state so the other tbot knows the updated state before choosing an action etc.
        
            state_active=tbot_active.dict_to_np_state(current_state,tbot_passive)  #active bots state tuple
            state_passive=tbot_passive.dict_to_np_state(current_state,tbot_active) #passive bots state tuple


# =============================================================================
#               passive tbot acts and does NOT learn
# =============================================================================
            action_P,action_items_P,action_params_P,action_string_P=self.choose_action(tbot_passive,epsilon,state_passive)

            through_api,next_state,reward=self.reward_prune(current_state,state_passive,state_active,action_P,action_items_P,tbot_passive,tbot_active) #reward won't be used 
        
            if through_api:
                success,next_state=api.execute_action(action_P,action_params_P,tbot_passive.name)
                reward=api.get_reward(current_state,action_P,next_state)

            R_cumulative_passive+=reward
            current_state=next_state # udpate current state for active tbot
        
            state_active=tbot_active.dict_to_np_state(current_state,tbot_passive)  #active bots state tuple
            state_passive=tbot_passive.dict_to_np_state(current_state,tbot_active) #passive bots state tuple

        R_cumulative[tbot_active.name].append(R_cumulative_active)
        R_cumulative[tbot_passive.name].append(R_cumulative_passive)
        print(R_cumulative)

        
        
        return q_values
    def task4(self,episodes):
        ''' train with 3 books for each tbot'''
        #pdb.set_trace()
        q_values = {}
        
        
        actions_json_file='/action_config.json'

        with open(self.root_path + actions_json_file) as json_file:
            try:
                self.action_reference = json.load(json_file, parse_float=float)
            except (ValueError, KeyError, TypeError):
                print "JSON error"
        
        
        
# =============================================================================
#       episode parameters
# =============================================================================
        #there are actually 2*episodes episodes, since there are two tbots
        episode_update=2 #the amount of episodes a tbot will train while the other tbots policy remains constant. must be a divisor of episodes
        episode_blocks=int(episodes/episode_update) #number of episode blocks, each episode block one tbot is updating their q table while the other only acts 
# =============================================================================
#       epsilon parameters & set penalty values 
# =============================================================================
        epsilon_initial=.95
        epsilon_decay=.002
        epsilon_calc= lambda epsilon_initial,epsilon_decay,i: max(0.05, epsilon_initial - epsilon_decay*i) 
        
        
        self.book_penalty=-100
        self.bump_penalty=-100
        
        
# =============================================================================
#         q tables initialized to zero 
# =============================================================================
        q1=np.zeros((7,7,4,2,2,2,2,2,2,5,9)) #(x,y, orientation, c1,c2,c3,c4,c5,c6,tbot_near,action) #c1,c2 will be zero if available, and one if picked up 
        q2=np.zeros((7,7,4,2,2,2,2,2,2,5,9))
        
# =============================================================================
#       Create Agents
# =============================================================================
        
        agent1_books=[1,2,3]
        agent2_books=[4,5,6]
        
        agent1=Agent('robot1',q1,agent1_books,more_books=True) 
        agent2=Agent('robot2',q2,agent2_books,more_books=True)
        
        tbot_list=[agent1,agent2]
        
        
        R_cumulative={agent1.name:[],agent2.name:[]}
        
# =============================================================================
#       acting and training 
# =============================================================================
        
        for i in range(episode_blocks):
          epsilon=epsilon_calc(epsilon_initial,epsilon_decay,i) #epsilon can only changes every episode block
          for tbot in tbot_list: #determines which tbot is learning, active updates table, passive does not
            tbot_active=tbot
            tbot_passive_set=set(tbot_list)-set([tbot])
            tbot_passive=tbot_passive_set.pop()
            for e in range(episode_update):#cycle through the episodes inside an episode block
              epsilon=epsilon_calc(epsilon_initial,epsilon_decay,i*episode_update+e)
              # a single episode
              api.reset_world()
              R_cumulative_active=0
              R_cumulative_passive=0
              initial_state=api.get_current_state()
              current_state=initial_state
              
              state_active=tbot_active.dict_to_np_state(current_state,tbot_passive)  #active bots state tuple
              state_passive=tbot_passive.dict_to_np_state(current_state,tbot_active) #passive bots state tuple
              #pdb.set_trace()
              print('episode_block {0} episode {1} for tbot {2}'.format(i,e, tbot_active.name))
              while not api.is_terminal_state(current_state):
                
                through_api=True # flag for going through API for an action, if False, then reward is given manually

# =============================================================================
#               active tbot acts and learns 
# =============================================================================
                #pick action for tbot_active
                #choose either random or exploit, according to epsilon=epsilon_calc(epsilon_initial, epsilon_decay, i)
# =============================================================================
#                 if state_active[0]>=5 or state_passive[0]>=5:
#                     pdb.set_trace()
# =============================================================================
                action_A,action_items_A,action_params_A,action_string_A=self.choose_action(tbot_active,epsilon,state_active) #selects action

                through_api,next_state,reward=self.reward_prune(current_state,state_active,state_passive,action_A,action_items_A,tbot_active,tbot_passive) #prunes by checking for invalid actions, in which case we don't run through environment_api 
                #pdb.set_trace()
                if through_api:
                  success,next_state=api.execute_action(action_A,action_params_A,tbot_active.name)
                  reward=api.get_reward(current_state,action_A,next_state)
                
                R_cumulative_active+=reward
                
                
                next_state_active=tbot_active.dict_to_np_state(next_state,tbot_passive)
                
                
                #update q_values of tbot_active ONLY
                
                state_action_idx=tuple(state_active)+tuple([tbot_active.idx_to_action.index(action_string_A)])
                tbot_active.q[state_action_idx]=(1-self.alpha)*tbot_active.q[state_action_idx]+self.alpha*(reward+self.gamma*max(tbot_active.q[next_state_active]))

                current_state=next_state # udpate current state so the other tbot knows the updated state before choosing an action etc.
                
                state_active=tbot_active.dict_to_np_state(current_state,tbot_passive)  #active bots state tuple
                state_passive=tbot_passive.dict_to_np_state(current_state,tbot_active) #passive bots state tuple
        
        
# =============================================================================
#               passive tbot acts and does NOT learn
# =============================================================================
                action_P,action_items_P,action_params_P,action_string_P=self.choose_action(tbot_passive,epsilon,state_passive)
        
                through_api,next_state,reward=self.reward_prune(current_state,state_passive,state_active,action_P,action_items_P,tbot_passive,tbot_active) #reward won't be used 
                
                if through_api:
                  success,next_state=api.execute_action(action_P,action_params_P,tbot_passive.name)
                  reward=api.get_reward(current_state,action_P,next_state)
        
                R_cumulative_passive+=reward
                current_state=next_state # udpate current state for active tbot
                
                state_active=tbot_active.dict_to_np_state(current_state,tbot_passive)  #active bots state tuple
                state_passive=tbot_passive.dict_to_np_state(current_state,tbot_active) #passive bots state tuple
        
              R_cumulative[tbot_active.name].append(R_cumulative_active)
              R_cumulative[tbot_passive.name].append(R_cumulative_passive)
              print(R_cumulative)
              print('on the {0}th episode is {1}'.format(i*episode_update+e, epsilon))
        np.save('q_table_r1_6.npy',agent1.q)
        np.save('q_table_r2_6.npy', agent2.q)
        
        import pickle
        with open("robot1_rewards_6.txt", "wb") as f:
            pickle.dump(R_cumulative['robot1'], f)
            
        with open("robot2_rewards_6.txt", "wb") as f:
            pickle.dump(R_cumulative['robot2'], f)
    def task5(self,episodes):
        
        '''for running the simulation after training'''
        q_values = {}
        # Your code here
        
        actions_json_file='/action_config.json'

        with open(self.root_path + actions_json_file) as json_file:
            try:
                self.action_reference = json.load(json_file, parse_float=float)
            except (ValueError, KeyError, TypeError):
                print "JSON error"
        
        self.book_penalty=-100
        self.bump_penalty=-100
        
        
# =============================================================================
#         q tables initialized to zero 
# =============================================================================
        #(x,y, orientation, c1,c2,tbot_near,action) #c1,c2 will be zero if available, and one if picked up 

        
        q1 = np.load('q_table_r1_6.npy')
        q2 = np.load('q_table_r2_6.npy')
# =============================================================================
#       Create Agents
# =============================================================================
        
        agent1_books=[1,2,3]
        agent2_books=[4,5,6]
        
        agent1=Agent('robot1',q1,agent1_books,more_books=True) 
        agent2=Agent('robot2',q2,agent2_books,more_books=True)
        
        tbot_list=[agent1,agent2]

        R_cumulative={agent1.name:[],agent2.name:[]}
        
# =============================================================================
#       acting 
# =============================================================================
        
        
      
        tbot_active=agent1
        tbot_passive=agent2
        epsilon=0
        # a single episode
        api.reset_world()
        R_cumulative_active=0
        R_cumulative_passive=0
        initial_state=api.get_current_state()
        current_state=initial_state
      
        state_active=tbot_active.dict_to_np_state(current_state,tbot_passive)  #active bots state tuple
        state_passive=tbot_passive.dict_to_np_state(current_state,tbot_active) #passive bots state tuple
        #pdb.set_trace()
      
        while not api.is_terminal_state(current_state):
        
            through_api=True # flag for going through API for an action, if False, then reward is given manually

# =============================================================================
#               active tbot acts and learns 
# =============================================================================
        #pick action for tbot_active
        #choose either random or exploit, according to epsilon=epsilon_calc(epsilon_initial, epsilon_decay, i)
# =============================================================================
#                 if state_active[0]>=5 or state_passive[0]>=5:
#                     pdb.set_trace()
# =============================================================================
            action_A,action_items_A,action_params_A,action_string_A=self.choose_action(tbot_active,epsilon,state_active) #selects action

            through_api,next_state,reward=self.reward_prune(current_state,state_active,state_passive,action_A,action_items_A,tbot_active,tbot_passive) #prunes by checking for invalid actions, in which case we don't run through environment_api 
            #pdb.set_trace()
            if through_api:
                success,next_state=api.execute_action(action_A,action_params_A,tbot_active.name)
                reward=api.get_reward(current_state,action_A,next_state)
        
            R_cumulative_active+=reward
        
        
            next_state_active=tbot_active.dict_to_np_state(next_state,tbot_passive)
        
        
            #update q_values of tbot_active ONLY
        
            state_action_idx=tuple(state_active)+tuple([tbot_active.idx_to_action.index(action_string_A)])

            current_state=next_state # udpate current state so the other tbot knows the updated state before choosing an action etc.
        
            state_active=tbot_active.dict_to_np_state(current_state,tbot_passive)  #active bots state tuple
            state_passive=tbot_passive.dict_to_np_state(current_state,tbot_active) #passive bots state tuple


# =============================================================================
#               passive tbot acts and does NOT learn
# =============================================================================
            action_P,action_items_P,action_params_P,action_string_P=self.choose_action(tbot_passive,epsilon,state_passive)

            through_api,next_state,reward=self.reward_prune(current_state,state_passive,state_active,action_P,action_items_P,tbot_passive,tbot_active) #reward won't be used 
        
            if through_api:
                success,next_state=api.execute_action(action_P,action_params_P,tbot_passive.name)
                reward=api.get_reward(current_state,action_P,next_state)

            R_cumulative_passive+=reward
            current_state=next_state # udpate current state for active tbot
        
            state_active=tbot_active.dict_to_np_state(current_state,tbot_passive)  #active bots state tuple
            state_passive=tbot_passive.dict_to_np_state(current_state,tbot_active) #passive bots state tuple

        R_cumulative[tbot_active.name].append(R_cumulative_active)
        R_cumulative[tbot_passive.name].append(R_cumulative_passive)
        print(R_cumulative)
        
    def task1(self, trajectories_json_file):
        q_values = {}
        # Your code here
        print(self.helper.get_all_actions())
        #print(self.helper.get_all_actions(2))
        api.execute_action('moveF', {}, 'robot1')
        #self.helper.execute_action('moveF', {}, 'robot2')
        
 
        return q_values  

if __name__ == "__main__":

    args = parser.parse_args()

    if args.task == 1:
        QLearning(args.task, headless=args.headless, sample=args.sample)
    elif args.task == 2 or args.task == 3 or args.task==4 or args.task==5:
        QLearning(args.task, headless=args.headless, episodes=args.episodes)