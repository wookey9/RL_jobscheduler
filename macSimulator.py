import random
from collections import defaultdict

import numpy as np
import time, os
import multiprocess as mp
import agent

actType = [(2,1),(2,2),(4,1),(4,2),(6,1),(6,2),(8,1),(8,2)]

class MacSimulator:
    def __init__(self):
        self.num_cell = 4
        self.num_core = 4
        self.num_action = 8
        self.cell_list = []

        self.Q = defaultdict(lambda: np.zeros(self.num_action))

    def getState(self):
        state = np.zeros((self.num_cell, 10))
        cellid = 0
        for cell in self.cell_list:
            state[cellid][0] = (int)(cell.totSch / 100)
            state[cellid][1] = (int)(cell.totTbs / 1000)
            state[cellid][2] = (int)(cell.cycle_proc[0] / 1000)
            state[cellid][3] = (int)(cell.cycle_proc[1] / 1000)
            state[cellid][4] = (int)(cell.cycle_proc[2] / 1000)

            cellid += 1
        return state
    def qLearning(self, action):
        self.agent = agent.Agent(self.Q, "q-learning")
        num_episodes = 100000
        for i_episode in range(1, num_episodes +1):
            state = self.getState()
            for i in range(500):
                action = self.agent.select_action(state)
                cellid = 0
                for cell in self.cell_list:
                    cell.setMaxSchPdu(actType[action][0])
                    cell.setNumCore(actType[action][1])
                    cellid += 1

                self.cell_list[0].run(int(time.time_ns() / 1000))
                self.cell_list[1].run(int(time.time_ns() / 1000))
                self.cell_list[2].run(int(time.time_ns() / 1000))
                self.cell_list[3].run(int(time.time_ns() / 1000))





        self.cell_list.clear()
        for cellid in range(self.num_cell):
            self.cell_list.append(Cell(int(np.random.rand()*256)))
        self.agent.select_action()
        for cell in self.cell_list:
            cell.run(int(time.time_ns() / 1000))

        return next_state,reward,done

    def reset(self):
        pass

class Cell:
    def __init__(self, num_ue, max_schpdu = 2, numcore = 1):
        self.max_schpdu = max_schpdu
        self.num_ue = num_ue
        self.ue_list = []
        self.maxTbs = 1000
        self.totTbs = 0
        self.totSch = 0
        self.cycle_proc = (0,0,0)
        self.cycle_slot_limit = 200000
        self.run_cnt = 0
        self.skip_cnt = 0
        self.num_core = numcore

        for uid in range(num_ue):
            if(uid < num_ue * 3 / 10) :
                self.ue_list.append(Ue(uid,10,True))
            else:
                self.ue_list.append(Ue(uid, 10, False))
    def setNumCore(self, numcore):
        self.num_core = numcore

    def setMaxSchPdu(self, max_schpdu):
        self.max_schpdu = max_schpdu

    def rbAllocation(self, tbs, mcsProcess):
        self.mcsCalculation(mcsProcess)
        time.sleep(tbs * 0.0001)

    def mcsCalculation(self, numProcess):
        time.sleep(numProcess * 0.002)

    def schedule(self, ue, leftTbs):
        schTbs = 0
        if ue.bo < leftTbs:
            schTbs = ue.bo
        else:
            schTbs = leftTbs

        procs = []
        for c in range(self.num_core):
            proc = mp.Process(target = self.rbAllocation, args = (schTbs / self.num_core ,10 / self.num_core,))
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()


        leftBo = ue.bo - schTbs
        ue.update_bo(leftBo, self.run_cnt, self.maxTbs)
        return schTbs

    def run(self, cycle_start):
        random.shuffle(self.ue_list)
        self.ue_list.sort(key=lambda ue: ue.bo, reverse=True)
        self.ue_list.sort(key=lambda ue: self.run_cnt - ue.updateTime, reverse=True)

        time.sleep(0.0001 * self.num_ue)
        schcnt = 0
        schTbsSum = 0

        for ue in self.ue_list:
            if((schcnt < self.max_schpdu) & (schTbsSum < self.maxTbs)):
                schTbsSum += self.schedule(ue, self.maxTbs - schTbsSum)
                schcnt += 1
            else:
                break

        self.run_cnt += 1

        cycle_end = int(time.time_ns() / 1000)
        cycle_proc = cycle_end - cycle_start

        if(cycle_proc >= self.cycle_slot_limit):
            self.skip_cnt += 1
        else:
            self.totSch += schcnt
            self.totTbs += schTbsSum

        cur_min, cur_max, cur_sum = self.cycle_proc
        if((cur_min > cycle_proc) | (cur_min == 0)):
            cur_min = cycle_proc
        if(cur_max < cycle_proc) :
            cur_max = cycle_proc
        cur_sum += cycle_proc
        self.cycle_proc = (cur_min, cur_max, cur_sum)

        print("[{}] sch : {}, acctbs : {}, curSch : {}, curTbs : {}, proc : {}, min : {}, max : {}, avg : {}, skip : {}".format(self.run_cnt, self.totSch, self.totTbs, schcnt, schTbsSum, cycle_proc, cur_min, cur_max, int(cur_sum / self.run_cnt), self.skip_cnt))

class Ue:
    def __init__(self, uid, bo, heavy) :
        self.uid = uid
        self.bo = bo
        self.updateTime = 1
        self.heavy = heavy
    def update_bo(self, leftBo, updateTime, maxTbs):
        if(leftBo <= 0):
            if(self.heavy):
                leftBo = int(np.random.rand() * maxTbs / 5) + 800
            else:
                leftBo = int(np.random.rand() * maxTbs / 10)
        self.bo = leftBo
        self.updateTime = updateTime


print("Number of processors : ", mp.cpu_count())
cell = Cell(256,12,4)

for i in range(500):
    cell.run(int(time.time_ns() / 1000))

for ue in cell.ue_list:
    if ue.updateTime == 1:
        print("Not scheduled UE : {}".format(ue.uid))