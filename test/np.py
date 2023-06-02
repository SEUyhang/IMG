import numpy as np
from gym import spaces

action1 = spaces.Box(low=-np.pi, high=+np.pi, shape=(2,),dtype=np.float32)
action2 = spaces.Box(low=-1.0, high=+1.0, shape=(2,),dtype=np.float32)
action = spaces.Tuple([action1,action2])
a1 = action1.sample()
a2 = action2.sample()
a = action.sample()
print('a1:  ', a1)
print('a2:  ', a2)
print('a:  ',a[0])