"""Explicit commissioning for the Mojia baud comparison, outside playback.

Register 6/55 behavior was verified on this head's STS3032 and SM45BL.
No goal, torque, limit, protection, PID or acceleration writes are made here.
Each write is issued once, then verified independently at a known baud.
"""
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from vivis.hardware.feetech.protocol import Bus, CorruptResponse, ReadTimeout, ProtocolError

BAUD_CODES={1000000:0,500000:1,250000:2}


def host_baud(bus,baud):
    bus.port.flush()
    bus.port.baudrate=baud
    bus.baud=bus.handler.baudrate=baud
    bus.handler.tx_time_per_byte=10000/baud
    time.sleep(.02)
    bus.port.reset_input_buffer()


def migrate(bus,baseline,old_baud,new_baud,journal_path):
    """Preflight every axis before any write; roll changed axes back on error.

    The journal is durable before each baud write, so interruption leaves a
    precise list of potentially changed IDs. Ordinary playback never calls this.
    """
    if old_baud not in BAUD_CODES or new_baud not in BAUD_CODES or old_baud==new_baud:
        raise ValueError('Invalid baud transition')
    if [s['id'] for s in baseline]!=list(range(2,37)):
        raise ValueError('Require complete ordered 35-axis baseline')
    journal={'from_baud':old_baud,'to_baud':new_baud,'events':[],
             'completed':False,'rolled_back':False}
    path=Path(journal_path)
    def save():
        tmp=path.with_suffix('.tmp')
        with tmp.open('w') as handle:
            handle.write(json.dumps(journal,indent=2)+'\n');handle.flush();os.fsync(handle.fileno())
        tmp.replace(path)
    def event(**data):journal['events'].append(data);save()
    def identity(sid,baud):
        host_baud(bus,baud)
        return readback(sid,0,9)
    def readback(sid,address,count):
        # Read-only retries tolerate a damaged reply, never repeat a write.
        for attempt in range(3):
            try:return bus._read_once(sid,address,count)
            except (CorruptResponse,ReadTimeout) as exc:
                event(id=sid,address=address,read_error=str(exc))
                if attempt==2:raise
                time.sleep(.01)
    def checked_write(sid,address,value):
        ack=bus.sdk.write1ByteTxRx(sid,address,value)
        event(id=sid,address=address,value=value,ack=list(ack))
        # Missing/damaged ACK is inconclusive: issue no second write, verify
        # the actual register instead. A valid servo fault remains an error.
        if ack[1] or ack[0] not in (0,-6,-7):raise ProtocolError(f'ID{sid} write {address} failed: {ack}')
    def short_read(sid,address,count):
        return b''.join(readback(sid,address+offset,min(8,count-offset)) for offset in range(0,count,8))
    def change(sid,source,destination,expected):
        found=identity(sid,source)
        if found!=expected:raise ProtocolError(f'ID{sid}: source identity/config mismatch')
        if readback(sid,55,1)!=b'\x01':raise ProtocolError(f'ID{sid}: lock must start at 1')
        checked_write(sid,55,0)
        if readback(sid,55,1)!=b'\x00':raise ProtocolError('Unlock readback mismatch')
        event(id=sid,phase='before_baud_write',source=source,destination=destination)
        checked_write(sid,6,BAUD_CODES[destination])
        time.sleep(.03)
        wanted=bytearray(expected);wanted[6]=BAUD_CODES[destination]
        if identity(sid,destination)!=bytes(wanted):raise ProtocolError(f'ID{sid}: destination readback mismatch')
        checked_write(sid,55,1)
        if readback(sid,55,1)!=b'\x01':raise ProtocolError('Relock readback mismatch')
        event(id=sid,phase='verified',baud=destination)
    save();host_baud(bus,old_baud)
    for s in baseline:
        sid=s['id'];expected=bytearray.fromhex(s['config_0_39_hex']);expected[6]=BAUD_CODES[old_baud]
        if s['model_raw']!=(10760 if 31<=sid<=35 else 521):raise ProtocolError('Unexpected model')
        # Bound the reply burst during commissioning at the slower baud too.
        # The entire configuration must still match, byte for byte.
        actual=short_read(sid,0,40)
        if actual!=bytes(expected) or readback(sid,55,1)!=b'\x01':
            raise ProtocolError(f'ID{sid}: baseline or lock mismatch')
    changed=[]
    try:
        for s in baseline:
            sid=s['id'];expected=bytearray.fromhex(s['config_0_39_hex'])[:9];expected[6]=BAUD_CODES[old_baud]
            changed.append(sid)  # Include the in-flight axis in rollback.
            change(sid,old_baud,new_baud,bytes(expected))
        host_baud(bus,new_baud)
        after=[Bus.snapshot(SimpleNamespace(read=short_read),s['id']) for s in baseline]
        for before,current in zip(baseline,after):
            cfg=bytearray.fromhex(before['config_0_39_hex']);cfg[6]=BAUD_CODES[new_baud]
            if current['config_0_39_hex']!=cfg.hex():raise ProtocolError('Configuration changed beyond baud')
            a=bytes.fromhex(before['state_40_70_hex']);b=bytes.fromhex(current['state_40_70_hex'])
            if any(a[i]!=b[i] for i in (0,1,6,7,8,9,15)):
                raise ProtocolError('Torque/acceleration/speed/torque-limit/lock changed')
        journal['after']=after;journal['completed']=True;save()
        return journal
    except BaseException as exc:
        journal['error']=str(exc);save();failures=[]
        for sid in reversed(changed):
            try:
                located=None
                for baud in (new_baud,old_baud):
                    try:found=identity(sid,baud);located=baud;break
                    except (ReadTimeout,CorruptResponse):pass
                if located is None:raise ProtocolError('Cannot locate axis for rollback')
                checked_write(sid,55,1)
                if located!=old_baud or found[6]!=BAUD_CODES[old_baud]:
                    change(sid,located,old_baud,found)
                if readback(sid,55,1)!=b'\x01':raise ProtocolError('Rollback lock mismatch')
            except BaseException as restore_error:failures.append({'id':sid,'error':str(restore_error)})
        host_baud(bus,old_baud)
        journal['rollback_failures']=failures;journal['rolled_back']=not failures;save()
        raise
