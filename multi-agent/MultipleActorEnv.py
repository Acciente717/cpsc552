import random
import torch
from PathPlanningEnv import PathPlanningEnv

class MultipleActorEnv():
    def __init__(self, *, env_num, height, width, obs_count, random_seed=None):
        assert env_num >= 1, "Cannot create less than 1 Envs"
        assert height > 0, "illegal height"
        assert width > 0, "illegal width"
        assert height * width > obs_count + 2, "illegal height, width, and obs_count combination"

        if random_seed is None:
            random_seed = 100
        self.env_num = env_num
        self.envs = []
        random.seed(random_seed)
        for _ in range(self.env_num):
            model_settings = {
                'height' : height,
                'width' : width,
                'obs_count' : obs_count,
                'random_seed' : random.randint(0, 2**16)}
            # print(model_settings)
            self.envs.append(PathPlanningEnv(**model_settings))
        self.actions = self.envs[0].actions
    
    def display(self):
        for i in range(self.env_num):
            print("{}-th env".format(i))
            self.envs[i].display()

    def reset(self):
        for env in self.envs:
            env.reset()
    
    def observation(self):
        batch_done = [env.done for env in self.envs]
        batch_states = [env.grid for env in self.envs if not env.done]
        batch_states = torch.stack(batch_states)
        print(type(batch_states), batch_states.shape)
        return batch_done, batch_states

    def step(self, mask, actions, early_stop=True, q_learning=False):
        assert len(mask)==self.env_num, "dimension of mask does not match env_num"
        assert len(actions)==self.env_num, "dimension of actions does not match env_num"
        
        rewards = [0]*self.env_num
        dones = [True]*self.env_num
        for i in range(self.env_num):
            if (mask[i]==1): 
                _, reward, done, _ = self.envs[i].step(actions[i])
                rewards[i] = reward
                dones[i] = done

        return rewards, dones
