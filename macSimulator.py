import numpy as np
import time, os
import multiprocess as mp

class MacSimulator:
    def __init__(self):
        self.num_cell = 4
        self.num_core = 4
        self.cell_list = []

    def step(self, action):
        self.cell_list.clear()
        for cellid in range(self.num_cell):
            self.cell_list.append(Cell(int(np.random.rand()*256), action[cellid]))
        for cell in self.cell_list:
            cell.run()

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
        self.cycle_slot_limit = 150000
        self.run_cnt = 0
        self.skip_cnt = 0
        self.num_core = numcore

        for uid in range(num_ue):
            if(uid < num_ue * 3 / 10) :
                self.ue_list.append(Ue(uid,10,True))
            else:
                self.ue_list.append(Ue(uid, 10, False))

    def rbAllocation(self, tbs):
        for i in range(tbs):
            time.sleep(0.01)

    def schedule(self, ue, leftTbs):
        schTbs = 0
        if ue.bo < leftTbs:
            schTbs = ue.bo
        else:
            schTbs = leftTbs

        procs = []
        for c in range(self.num_core):
            proc = mp.Process(target = self.rbAllocation, args = (int(schTbs / 100 / self.num_core),))
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()


        leftBo = ue.bo - schTbs
        ue.update_bo(leftBo, self.run_cnt, self.maxTbs)
        return schTbs

    def run(self, cycle_start):
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
            print("[{}] sch : {}, acctbs : {}, curSch : {}, curTbs : {}, proc : {}, skip : {}".format(self.run_cnt, self.totSch, self.totTbs, schcnt, schTbsSum, cycle_proc, self.skip_cnt))
        else:
            cur_min, cur_max, cur_sum = self.cycle_proc
            self.totSch += schcnt
            self.totTbs += schTbsSum

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
cell = Cell(256,4,1)

for i in range(1000):
    cell.run(int(time.time_ns() / 1000))

for ue in cell.ue_list:
    if ue.updateTime == 1:
        print("Not scheduled UE : {}".format(ue.uid))