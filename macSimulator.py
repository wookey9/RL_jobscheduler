import json
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import time, os
import multiprocess as mp
import agent

actType = [(2,1),(4,2)]

class MacSimulator:
    def __init__(self):
        self.num_cell = 4
        self.num_core = 4
        self.num_ue = 256
        self.num_action_cell = 2
        self.num_action = 16
        self.cell_list = []
        self.ue_list = []
        self.cell_cycle_stat = [(0,0,0,0,0) for i in range(self.num_cell)]
        self.cell_tput_stat = [(0,0) for i in range(self.num_cell)]
        self.step_cnt = 10
        self.mode = "q-learning"
        self.Q = defaultdict(lambda: np.zeros(self.num_action))
        if os.path.isfile('q-data.json'):
            with open('q-data.json', 'r') as f:
                self.Q = json.load(f)
        self.rewardHistory = []

    def setMode(self, mode):
        self.mode = mode

    def getState(self):
        rows = self.num_cell * 8
        state = [0 for i in range(rows)]
        cellid = 0
        for cell in self.cell_list:
            offset = cellid * 8
            state[offset + 0] = (int)(self.cell_tput_stat[cell.ccid][0] / 100)
            state[offset + 1] = (int)(self.cell_tput_stat[cell.ccid][1])
            state[offset + 2] = (int)(self.cell_cycle_stat[cell.ccid][0] / 1000)
            state[offset + 3] = (int)(self.cell_cycle_stat[cell.ccid][1] / 1000)
            state[offset + 4] = (int)(self.cell_cycle_stat[cell.ccid][2] / 1000)
            state[offset + 5] = (int)(self.cell_cycle_stat[cell.ccid][3] / self.step_cnt / 1000)
            state[offset + 6] = (int)(self.cell_cycle_stat[cell.ccid][4])
            state[offset + 7] = (int)(cell.num_ue / 10)
            cellid += 1
        return tuple(state)

    def getReward(self):
        reward = 0
        for cell in self.cell_list:
            reward += (int(self.cell_tput_stat[cell.ccid][0] / 1000) + self.cell_tput_stat[cell.ccid][1] * 100)
        return reward

    def initStat(self):
        self.cell_cycle_stat = [(0,0,0,0,0) for i in range(self.num_cell)]
        self.cell_tput_stat = [(0,0) for i in range(self.num_cell)]

    def updateUeState(self, curTime):
        for cell in self.cell_list:
            while not (cell.sched_ue.empty()):
                uid, schBo = cell.sched_ue.get()
                self.ue_list[uid].update_bo(self.ue_list[uid].bo - schBo, curTime, cell.maxTbs)

    def updateCellStat(self, step):
        for cell in self.cell_list:
            while not (cell.sch_result.empty()):
                proc, tbs, cnt, skip = cell.sch_result.get()

                stat_tbs, stat_cnt = self.cell_tput_stat[cell.ccid]
                cur_proc, cy_min, cy_max, cy_sum, skipCnt = self.cell_cycle_stat[cell.ccid]

                self.cell_tput_stat[cell.ccid] = (stat_tbs + tbs, stat_cnt + cnt)

                if((cy_min > proc) | (cy_min == 0)):
                    cy_min = proc
                if(cy_max < proc) :
                    cy_max = proc
                cy_sum += proc
                skipCnt += (skip == True)
                self.cell_cycle_stat[cell.ccid] = (proc, cy_min, cy_max, cy_sum, skipCnt)

    def applyAction(self, state):
        action = self.agent.select_action(state)
        offset = action

        for cellid in range(self.num_cell):
            sch_pdu = actType[offset % self.num_action_cell][0]
            core_num = actType[offset % self.num_action_cell][1]
            self.cell_list[cellid].setMaxSchPdu(sch_pdu)
            self.cell_list[cellid].setNumCore(core_num)
            offset = int(offset / self.num_action_cell)
        return action

    def qLearning(self):
        self.agent = agent.Agent(self.Q, self.num_action, self.mode)
        num_episodes = 100
        for i_episode in range(1, num_episodes + 1):

            self.cell_list.clear()
            self.ue_list.clear()

            heavyCc = np.random.randint(0, self.num_cell, size = 1)

            for uid in range(self.num_ue):
                if(np.random.rand() < 0.5):
                    if(np.random.rand() < 0.3):
                        self.ue_list.append(Ue(uid, 10, True, np.random.randint(0, self.num_cell, size=1)))
                    else:
                        self.ue_list.append(Ue(uid, 10, False, np.random.randint(0, self.num_cell, size=1)))
                else:
                    if(np.random.rand() < 0.7):
                        if (np.random.rand() < 0.3):
                            self.ue_list.append(Ue(uid, 10, True, heavyCc))
                        else:
                            self.ue_list.append(Ue(uid, 10, False, heavyCc))
                    else:
                        if (np.random.rand() < 0.5):
                            self.ue_list.append(Ue(uid, 10, True, np.random.randint(0, self.num_cell, size=1)))
                        else:
                            self.ue_list.append(Ue(uid, 10, False, np.random.randint(0, self.num_cell, size=1)))
            for cellid in range(self.num_cell):
                self.cell_list.append(Cell(cellid, list(filter(lambda x: x.ccid == cellid, self.ue_list))))

            state = self.getState()
            action = self.applyAction(state)
            self.initStat()

            for i in range(self.step_cnt):
                coreCnt = 0
                startTime = int(time.time_ns() / 1000)
                procs = []
                procs_coreNum = []
                procCnt = 0
                for Cell in self.cell_list:
                    if(coreCnt + cell.num_core <= self.num_core):
                        coreCnt += cell.num_core
                        procCnt += 1
                        proc = mp.Procss(target = cell.run, args = (startTime, i))
                        procs.append(proc)
                        procs_coreNum.append(cell.num_core)
                        proc.start()
                    else:
                        while True:
                            any_proc_done = False
                            procId = 0
                            for proc in procs:
                                if proc.is_alive() == False:
                                    any_proc_done = True
                                    break
                                procId += 1
                            if(procId < procCnt):
                                procs.remove(proc)
                                coreCnt -= procs_coreNum[procId]
                                procCnt -= 1
                                del procs_coreNum[procId]

                                if(any_proc_done & (coreCnt + cell.num_core <= self.num_core)):
                                    coreCnt += cell.num_core
                                    proc = mp.Process(target = cell.run, args = (startTime, i))
                                    procs.append(proc)
                                    procs_coreNum.append(cell.num_core)
                                    proc.start()
                                    break

                for proc in procs:
                    proc.join()

                self.updateUeState(i)
                self.updateCellStat(i)

            next_state = self.getState()
            reward = self.getReward()
            self.agent.step(state, action, reward, next_state, (i_episode % 100 == 0))
            state = next_state
            self.rewardHistory.append(reward)
            for cell in self.cell_list:
                print("[{}] [{}] [{}]  Action[{}/{}] State[{}/{}/{}/{}/{}/{}/{}/{}] Reward[{}]".format(i_episode, cell.ccid, cell.num_ue, cell.num_core, cell.max_schpdu,
                                                                                                       next_state[cell.ccid * 8 + 0],
                                                                                                       next_state[cell.ccid * 8 + 1],
                                                                                                       next_state[cell.ccid * 8 + 2],
                                                                                                       next_state[cell.ccid * 8 + 3],
                                                                                                       next_state[cell.ccid * 8 + 4],
                                                                                                       next_state[cell.ccid * 8 + 5],
                                                                                                       next_state[cell.ccid * 8 + 6],
                                                                                                       next_state[cell.ccid * 8 + 7],
                                                                                                       reward
                                                                                                       ))

class Cell:
    def __init__(self, ccid, ue_list, max_schpdu = 2, numcore = 1):
        self.ccid = ccid
        self.max_schpdu = max_schpdu
        self.num_ue = len(ue_list)
        self.maxTbs = 1000
        self.cycle_slot_limit = 200000
        self.num_core = numcore
        self.cur_proc_cycle = 0

        self.ue_list = ue_list

        self.sched_ue = mp.Queue()
        self.sch_result = mp.Queue()

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

    def run(self, cycle_start, curTime):
        random.shuffle(self.ue_list)
        self.ue_list.sort(key=lambda ue: ue.bo, reverse=True)
        self.ue_list.sort(key=lambda ue: curTime - ue.updateTime, reverse=True)

        time.sleep(0.0001 * self.num_ue)
        schcnt = 0
        schTbsSum = 0

        for ue in self.ue_list:
            if((schcnt < self.max_schpdu) & (schTbsSum < self.maxTbs)):
                tbs = self.schedule(ue, self.maxTbs - schTbsSum)
                self.sched_ue.put((ue.uid, tbs))
                schTbsSum += tbs
                schcnt += 1
            else:
                break

        cycle_end = int(time.time_ns() / 1000)
        cycle_proc = cycle_end - cycle_start

        if(cycle_proc >= self.cycle_slot_limit):
            schcnt = 0
            schTbsSum = 0

        self.sch_result.put((cycle_proc, schTbsSum, schcnt, cycle_proc >= self.cycle_slot_limit))
        #print("[{}] sch : {}, acctbs : {}, curSch : {}, curTbs : {}, proc : {}, min : {}, max : {}, avg : {}, skip : {}".format(self.run_cnt, self.totSch, self.totTbs, schcnt, schTbsSum, cycle_proc, cur_min, cur_max, int(cur_sum / self.run_cnt), self.skip_cnt))

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
        self.bo = leftBo
        self.updateTime = updateTime


print("Number of processors : ", mp.cpu_count())

macSimulator = MacSimulator()
while True:
    print("1. q-learning")
    print("2. test")
    mode = int(input("select : "))
    if(mode == 1):
        macSimulator.setMode("q-learning")
        macSimulator.qLearning()
        with open('q-date.json', 'w') as f:
            json.dump(macSimulator.Q, f)
        plt.plot(macSimulator.rewardHistory)
        plt.show()
    else:
        macSimulator.setMode("test-mode")
        macSimulator.qLearning()