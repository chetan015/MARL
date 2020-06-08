#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:56:01 2019

@author: cse-571
"""

import numpy as np
import pdb 



#q table is (x,y,c1,c2,tbot_near,num_actions)

class Agent:
    def __init__(self,name,q,books,more_books=False):
        
        self.name=name
        self.q=q
        self.books=books
        self.c=0 # 0 means that the coin is not picked up 
        self.more_books=more_books
        #self.reward_key=reward_key
        #self.reward_locs=grid_rewards
        
        self.o_to_idx={'NORTH':0,'EAST':1, 'SOUTH':2, 'WEST': 3} #orientation to idx in state
        self.pos_to_idx=lambda n:int(n/.5)
        
        self.idx_to_action=['moveF', 'TurnCW', 'TurnCCW', 'pick book_1', 'pick book_2']
        if self.more_books:
            self.idx_to_action=['moveF', 'TurnCW', 'TurnCCW', 'pick book_1', 'pick book_2','pick book_3','pick book_4','pick book_5','pick book_6']
        #self.bug_count=0
        
        
    def dict_to_np_state(self,d,other_bot):
        #returns state index for numpy array q table 
        c1x=d['book_1']['x']
        c1y=d['book_1']['y']
        c1_placed=d['book_1']['placed']
        c1_idx=int(c1_placed)
        
        c2x=d['book_2']['x']
        c2y=d['book_2']['y']
        c2_placed=d['book_2']['placed']
        c2_idx=int(c2_placed)
        
        if self.more_books:
            
            c1x=d['book_1']['x']
            c1y=d['book_1']['y']
            c1_placed=d['book_1']['placed']
            c1_idx=int(c1_placed)
            
            c2x=d['book_2']['x']
            c2y=d['book_2']['y']
            c2_placed=d['book_2']['placed']
            c2_idx=int(c2_placed)
            
            c3x=d['book_3']['x']
            c3y=d['book_3']['y']
            c3_placed=d['book_3']['placed']
            c3_idx=int(c3_placed)
            
            c4x=d['book_4']['x']
            c4y=d['book_4']['y']
            c4_placed=d['book_4']['placed']
            c4_idx=int(c4_placed)
            
            c5x=d['book_5']['x']
            c5y=d['book_5']['y']
            c5_placed=d['book_5']['placed']
            c5_idx=int(c5_placed)
            
            c6x=d['book_6']['x']
            c6y=d['book_6']['y']
            c6_placed=d['book_6']['placed']
            c6_idx=int(c6_placed)
        
        
        rx=self.pos_to_idx(d[self.name]['x'])
        ry=self.pos_to_idx(d[self.name]['y'])
        ro=self.o_to_idx[d[self.name]['orientation']]
        
        
        rx_other=self.pos_to_idx(d[other_bot.name]['x'])
        ry_other=self.pos_to_idx(d[other_bot.name]['y'])
        ro_other=self.o_to_idx[d[other_bot.name]['orientation']]
        #print(c1x,c1y,c2x,c2y)
# =============================================================================
#         if (c1x < 0 or c1y < 0 or c2x < 0 or c2y < 0) and self.bug_count==0:
#             self.bug_count+=1
#             pdb.set_trace()
# =============================================================================
# =============================================================================
#         if rx < 0 or ry < 0 or rx_other < 0 or ry_other < 0:
#             pdb.set_trace()
# =============================================================================
# =============================================================================
#         if rx>=5 or ry>=5:
#             pdb.set_trace()
# =============================================================================
        
        tbot_near=self.tbot_near([rx,ry],[rx_other,ry_other])
        
        
        #pdb.set_trace()
        
        state=(rx,ry,ro,c1_idx,c2_idx,tbot_near)
        #      x, y , coin,  other coin, tbot_near
        if self.more_books:
            state=(rx,ry,ro,c1_idx,c2_idx,c3_idx,c4_idx,c5_idx,c6_idx,tbot_near)
        return state 
        
    def tbot_near(self,r1,r2):
        #r1=[r1x,r1y] etc. robot x-y coordinates 
        #0,1,2,3 means that another tbot is above, to the right, below, or to the left
        #4 means that there is no bot near 
        #pdb.set_trace()
        if abs(r1[0]-r2[0])+abs(r1[1]-r2[1])==1:
            if r1[1]>r2[1]:
                return 2
            if r2[1]>r1[1]:
                return 0
            if r1[0]>r2[0]:
                return 3
            if r2[0]>r1[0]:
                return 1
            
        else:
            return 4
            
