"""Read short feedback fields only in the slack between motor commands.

One thread owns the half-duplex bus. A partial sample never updates freshness;
an invalid field discards that sample. The caller validates each complete sample.
"""
import time

from .cancellation import OperatorStop
from .protocol import CorruptResponse, ReadTimeout


class BudgetedFeedback:
    fields=((56,2),(60,2),(62,2),(65,1))

    def __init__(self,bus,ids,timeout_ms=3,priority_ids=(23,32),priority_period_s=.08):
        if not ids or len(set(ids))!=len(ids) or not set(priority_ids)<=set(ids):
            raise ValueError('Invalid feedback axes')
        if not 2<=timeout_ms<=5 or not .05<=priority_period_s<=.2:
            raise ValueError('Invalid feedback timing')
        self.bus=bus;self.ids=list(ids);self.timeout_s=timeout_ms/1000
        self.priority_ids=priority_ids;self.period=priority_period_s
        self.sid=None;self.parts=[];self.started=None
        self.last_attempt={}

    def step(self,deadline,clocks,on_sample,on_error,should_stop=lambda:False):
        reads=completed=0
        # Eight short requests can complete two samples. A timeout consumes its
        # reservation; there is no sleep or immediate retry after that timeout.
        while reads<8 and time.monotonic()+self.timeout_s+.00025<=deadline:
            if should_stop():raise OperatorStop('Operator stop during scheduled feedback')
            now=time.monotonic()
            if self.sid is None:
                # Failed samples do not refresh `clocks`. Track service separately
                # so one damaged axis cannot monopolize every subsequent tick.
                # Apply priority cadence to attempts too, including failed ones.
                def serviced(sid):return max(clocks[sid],self.last_attempt.get(sid,clocks[sid]))
                due=[sid for sid in self.priority_ids if now-serviced(sid)>=self.period]
                self.sid=min(due or self.ids,key=serviced)
                self.started=now
            sid=self.sid;address,count=self.fields[len(self.parts)]
            reads+=1
            try:data=self.bus.read(sid,address,count)
            except (CorruptResponse,ReadTimeout) as exc:
                self.last_attempt[sid]=time.monotonic()
                self.sid=None;self.parts=[];self.started=None
                on_error(sid,exc)
                break
            self.parts.append(data)
            if len(self.parts)==len(self.fields):
                position,load,sensors,status=self.parts
                state=dict(position=int.from_bytes(position,'little'),load_raw=int.from_bytes(load,'little'),
                           voltage_raw=sensors[0],temperature_c=sensors[1],status_register=status[0])
                started=self.started
                self.last_attempt[sid]=time.monotonic()
                self.sid=None;self.parts=[];self.started=None
                on_sample(sid,state,started)
                completed+=1
        return dict(reads=reads,completed=completed,partial_id=self.sid)
