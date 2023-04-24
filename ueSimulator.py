import numpy as np
class Ue:
    def __init__(self, uid, bo, heavy, ccid) :
        self.uid = uid
        self.bo = bo
        self.updateTime = 1
        self.heavy = heavy
        self.ccid = ccid
    def update_bo(self, leftBo, updateTime, maxTbs):
        if(leftBo <= 0):
            if(self.heavy):
                leftBo = int(np.random.rand() * maxTbs / 5) + 800
            else:
                leftBo = int(np.random.rand() * maxTbs / 10)
        else:
            leftBo = 0
        self.bo = leftBo
        self.updateTime = updateTime
