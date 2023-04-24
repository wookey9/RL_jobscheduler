import random
import time, os
import multiprocess as mp
import numpy as np
import ueSimulator
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
        time.sleep(tbs * 0.000001)

    def mcsCalculation(self, numProcess):
        time.sleep(numProcess * 0.00002)

    def schedule(self, ue, leftTbs):
        schTbs = 0
        if ue.bo < leftTbs:
            schTbs = ue.bo
        else:
            schTbs = leftTbs

        procs = []
        for c in range(self.num_core):
            proc = mp.Process(target = self.rbAllocation, args = (schTbs / self.num_core, 10 / self.num_core,))
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()

        return schTbs

    def run(self, cycle_start, curTime):
        random.shuffle(self.ue_list)
        self.ue_list.sort(key=lambda ue: ue.bo, reverse=True)
        self.ue_list.sort(key=lambda ue: curTime - ue.updateTime, reverse=True)

        time.sleep(0.000001 * self.num_ue)
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
        self.cur_proc_cycle = cycle_proc

        if(cycle_proc >= self.cycle_slot_limit):
            schcnt = 0
            schTbsSum = 0

        self.sch_result.put((cycle_proc, schTbsSum, schcnt, cycle_proc >= self.cycle_slot_limit))
        #print("[{}] sch : {}, acctbs : {}, curSch : {}, curTbs : {}, proc : {}, min : {}, max : {}, avg : {}, skip : {}".format(self.run_cnt, self.totSch, self.totTbs, schcnt, schTbsSum, cycle_proc, cur_min, cur_max, int(cur_sum / self.run_cnt), self.skip_cnt))

