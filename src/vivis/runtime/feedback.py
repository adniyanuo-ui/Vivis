"""Read short feedback fields only in the slack between motor commands.

One thread owns the half-duplex bus. A partial sample never updates freshness;
an invalid field discards that sample. The caller validates each complete sample.
"""
import time

from vivis.hardware.feetech.protocol import CorruptResponse, ReadTimeout


class BudgetedFeedback:
    fields=((56,2),(60,2),(62,2),(65,1))

    def __init__(self,bus,ids,timeout_ms=3,priority_ids=(23,32),priority_period_s=.08,max_feedback_age_s=1.,max_requests_per_step=8):
        if not ids or len(set(ids))!=len(ids) or not set(priority_ids)<=set(ids):
            raise ValueError('Invalid feedback axes')
        if not 2<=timeout_ms<=5 or not .05<=priority_period_s<=.2:
            raise ValueError('Invalid feedback timing')
        if type(max_feedback_age_s) not in (int,float) or not 0<max_feedback_age_s<=1:
            raise ValueError('Invalid feedback age deadline')
        if type(max_requests_per_step) is not int or not 1<=max_requests_per_step<=16:
            raise ValueError('Invalid feedback request budget')
        self.max_requests_per_step=max_requests_per_step
        self.bus=bus;self.ids=list(ids);self.timeout_s=timeout_ms/1000
        self.priority_ids=priority_ids;self.period=priority_period_s
        self.sid=None;self.parts=[];self.started=None
        self.last_attempt={}
        # Reserve the final quarter of the existing freshness window for
        # recovery. This changes service order, never the watchdog deadline.
        self.urgent_age_s=.75*max_feedback_age_s

    def step(self,deadline,clocks,on_sample,on_error,should_stop=lambda:False):
        reads=completed=0;selected=[];urgent_selected=[];failed=set()
        # A bounded batch can complete several samples. A timeout consumes its
        # reservation; there is no sleep or immediate retry after that timeout.
        while reads<self.max_requests_per_step and time.monotonic()+self.timeout_s+.00025<=deadline:
            if should_stop():raise RuntimeError('Operator stop during scheduled feedback')
            now=time.monotonic()
            if self.sid is None:
                # Valid feedback and attempted service are separate clocks.
                # A failed priority axis must not monopolize every next tick.
                def serviced(sid):return max(clocks[sid],self.last_attempt.get(sid,clocks[sid]))
                available=[sid for sid in self.ids if sid not in failed]
                if not available:break
                urgent=[sid for sid in available if now-clocks[sid]>=self.urgent_age_s]
                if urgent:
                    # Fair service still applies when several axes are urgent.
                    # A damaged axis retains its old VALID clock, but its fresh
                    # attempt must not block all the other approaching deadlines.
                    self.sid=min(urgent,key=lambda sid:(serviced(sid),clocks[sid]))
                    urgent_selected.append(self.sid)
                else:
                    due=[sid for sid in self.priority_ids if sid in available and now-serviced(sid)>=self.period]
                    self.sid=min(due or available,key=serviced)
                selected.append(self.sid)
                self.started=now
            sid=self.sid;address,count=self.fields[len(self.parts)]
            reads+=1
            try:data=self.bus.read(sid,address,count)
            except (CorruptResponse,ReadTimeout) as exc:
                self.last_attempt[sid]=time.monotonic()
                self.sid=None;self.parts=[];self.started=None
                on_error(sid,exc)
                # Never retry this axis within this tick. Other axes may use
                # the remaining bounded slot, rather than lose a whole30Hz tick.
                failed.add(sid)
                continue
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
        return dict(reads=reads,completed=completed,partial_id=self.sid,
                    selected_ids=selected,urgent_ids=urgent_selected,failed_ids=sorted(failed))
