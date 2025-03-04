
import numpy as np
import gymnasium as gym
import re

from typing import Optional, Union

from vsagym.wrappers.minigrid_wrappers.minigrid_view_wrapper import SSPMiniGridViewWrapper

from vsagym.spaces.ssp_box import SSPBox
from vsagym.spaces.ssp_discrete import SSPDiscrete
from vsagym.spaces.ssp_sequence import SSPSequence
from vsagym.spaces.ssp_dict import SSPDict

from gymnasium.spaces import Box, Discrete

    
class SSPMiniGridMissionWrapper(SSPMiniGridViewWrapper):
    def __init__(
        self,
        env: gym.Env,
        ssp_space: Optional[Union[SSPBox, SSPDiscrete, SSPSequence, SSPDict]] = None,
        shape_out: Optional[int] = None,
        obj_encoding: str = 'allbound',
        **kwargs
    ):
        
        super().__init__(env, ssp_space, shape_out, obj_encoding, **kwargs)
        self.observation_space["image"] = Box(low=-np.ones(2*self.shape_out), high=np.ones(2*self.shape_out))

        self.go_to_patterns = ["GO TO THE ", "GO TO A ", "GET A ", "GO GET A ", "FETCH A ", "GO FETCH A ",
                 "YOU MUST FETCH A"]
        self.pick_up_patterns = ["PICK UP THE ", "PICK UP A "]
        self.put_somewhere_patterns = ["PUT THE .* NEXT TO THE ", "PUT THE .* NEAR THE "]  # for BabyAI-PutNext
        self.open_something_patterns = ["OPEN THE ", "OPEN A ", "UNLOCK THE "]

        self.connector_words = ["AND", "THEN", "AFTER YOU"]

    
    def _encode_words(self, words):
        O_w = self.sp_space.name_to_vector['I'].copy()
        for w in words:
            w = w.strip()
            if w in self.color_map.keys():
                O_w += self.sp_space.bind(O_w, self.sp_space.name_to_vector[w].copy())
            elif w in self.obj_map.keys():
                O_w += self.sp_space.bind(O_w, self.sp_space.name_to_vector[w].copy())
            elif w in self.state_map.keys():
                O_w += self.sp_space.bind(O_w, self.sp_space.name_to_vector[w].copy())
            # elif w in self.locations:
            #     O_w +=  O_w * self.vocab[w]
            elif w not in [' ', 'IN', 'OF', 'YOU', 'ON', 'YOUR']:
                break
        return O_w
     
    def _encode_submission(self, mission):
        O = np.zeros((1,self.shape_out))
        for instruction in self.go_to_patterns:
            if re.match(".*" + instruction + ".*", mission):
                words = mission.partition(instruction)[2].split(" ")
                O += self.sp_space.bind(self.sp_space.name_to_vector['GO_TO'].copy(),
                                        self._encode_words(words))

        for instruction in self.pick_up_patterns:
            if re.match(".*" + instruction + ".*", mission):
                words = mission.partition(instruction)[2].split(" ")
                O += self.sp_space.bind(self.sp_space.name_to_vector['PICK_UP'].copy(),
                                        self._encode_words(words))

        for instruction in self.put_somewhere_patterns:
            if re.match(instruction, mission):
                put_what = re.search(instruction.replace('.*','(.*?)'), mission).group(1).split(" ")
                put_where = mission.partition(instruction[-12:])[2].split(" ")
                O += self.sp_space.bind(self.sp_space.name_to_vector['PUT'].copy(),
                                        self._encode_words(put_what))
                O += self.sp_space.bind(self.sp_space.name_to_vector['NEXT_TO'].copy(),
                                        self._encode_words(put_where))

        for instruction in self.open_something_patterns:
            if re.match(".*" + instruction + ".*", mission):
                words = mission.partition(instruction)[2].split(" ")
                O += self.sp_space.bind(self.sp_space.name_to_vector['OPEN'].copy(),
                                        self._encode_words(words))
            
        if re.match("GET THE .* KEY FROM THE .* ROOM, UNLOCK THE .* DOOR AND GO TO THE GOAL", mission): #for MiniGrid-LockedRoom-v0
            # lockedroom_color = re.search("GET THE (.*?) KEY", mission).group(1)
            keyroom_color = re.search("FROM THE (.*?) ROOM", mission).group(1)
            # door_color = re.search("UNLOCK THE (.*?) DOOR", mission).group(1)
            O += self.sp_space.bind(self.sp_space.name_to_vector['GO_TO'].copy(),
                                    self.sp_space.name_to_vector['DOOR'].copy(),
                                    self._encode_words(keyroom_color))
        return O.flatten()

    def _encode_mission(self, mission):
        mission_sp = np.zeros(self.shape_out)
        mission = mission.upper()
        submissions = re.split('|'.join(re.escape(term) for term in self.connector_words), mission)
        for m in submissions:
            mission_sp += self._encode_submission(m)
        return mission_sp

    def observation(self, obs):
        img = obs['image']
        
        agt_ssp = self._encode_agent_pos().flatten()
        has_sp = self._encode_carry()
        view_sp = self._encode_view(img)

        mission_sp = self._encode_mission(obs['mission'])

        vsa_encoding = np.hstack([agt_ssp + has_sp + view_sp,
                                  mission_sp]).flatten()
        
        return {
            'mission': obs['mission'],
            'image': vsa_encoding/np.linalg.norm(vsa_encoding),
        }
        # return {
        #     'mission': mission_sp.flatten(),
        #     'image': (agt_ssp + has_sp + view_sp).flatten()
        # }




class PrepMiniGridMissionWrapper(SSPMiniGridMissionWrapper):
    def __init__(
        self,
        env: gym.Env,
        ssp_space: Optional[Union[SSPBox, SSPDiscrete, SSPSequence, SSPDict]] = None,
        shape_out: Optional[int] = None,
        **kwargs
    ):
        
        super().__init__(env, ssp_space, shape_out, **kwargs)
        n_ssp_dims = (2 + self.view_width*self.view_height)
        self.observation_space['image'] = Box(low=-np.ones(n_ssp_dims*self.shape_out+self.shape_in),
                                     high=np.max([env.unwrapped.width, env.unwrapped.height])*np.ones(n_ssp_dims*self.shape_out+self.shape_in),
                                              dtype=self.ssp_obs_space.dtype)
        

    def _encode_view(self, img):
        obj_poss = []
        obj_sps = []
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                obj = self.idx_to_object[img[i, j, 0]]
                color = self.idx_to_color[img[i, j, 1]]
                state = self.idx_to_state[img[i, j, 2]]
                obj_pos = self._get_grid_pos(i,j)
                if obj in self.notice_objs:
                    # obj_ssp = self.ssp_pos_grid[i, j, :]
                    obj_name = self.obj_map[obj] # maps b/c some types are treated as NULL or I. TODO: maybe map right from idx to SP names without middle step
                    col_name = self.color_map[color]
                    state_name = self.state_map[state]

                    obj_sp = self._encode_object(obj_name, col_name, state_name)
                    obj_sps.append(obj_sp.flatten())
                else:
                    obj_sps.append(self.sp_space.name_to_vector['NULL'].flatten())
                obj_poss.append(obj_pos)
        return obj_poss, obj_sps


    def observation(self, obs):
        img = obs['image']
        agt_pt = np.array([self.env.unwrapped.agent_pos[0],
                            self.env.unwrapped.agent_pos[1],
                            self.env.unwrapped.agent_dir
                             ])
        has_vector = self._encode_carry()
        obj_poss, obj_sps = self._encode_view(img)
        mission_vector = self._encode_mission(obs['mission'])
        #TODO: pass obj_poss so that the MiniGrid feature extractors support global view too
        return {
            'mission': obs['mission'],
            'image': np.hstack([agt_pt, np.array(obj_sps).flatten(), has_vector.flatten(),
                                mission_vector.flatten()])
        }
        

    
    
