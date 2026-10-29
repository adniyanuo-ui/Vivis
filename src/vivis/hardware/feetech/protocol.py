"""Feetech magnetic-encoder packet transport. No EEPROM or torque-enable writes.
References: Feetech FT-SCS protocol and the pinned SMS/STS SDK.
"""
import struct
import time

# Verified FT-SCS sensor status bits: voltage, temperature, current and load.
# Encoder/unknown bits are deliberately excluded from deferred handling.
DEFERRED_SENSOR_ALARMS=0x01|0x04|0x08|0x20

class ProtocolError(RuntimeError): pass
class CorruptResponse(ProtocolError): pass
class ReadTimeout(ProtocolError): pass
class ServoStatusError(ProtocolError):
    def __init__(self,sid,status):
        self.sid=sid;self.status=status
        super().__init__(f'Servo {sid}: status 0x{status:02x}')

def packet(servo_id, instruction, parameters=b''):
    if not 0<=servo_id<=254 or len(parameters)>253: raise ValueError('Packet bounds')
    body=bytes([servo_id,len(parameters)+2,instruction])+bytes(parameters)
    return b'\xff\xff'+body+bytes([(~sum(body))&255])

def decode_status(raw,expected_id,expected_size):
    if len(raw)<6 or raw[:2]!=b'\xff\xff' or raw[2]!=expected_id:
        raise ProtocolError('Bad response header/id')
    if raw[3]!=len(raw)-4 or len(raw)!=expected_size+6 or sum(raw[2:])&255!=255:
        raise CorruptResponse(f'Bad response size/checksum: id={expected_id} expected_data={expected_size} raw={raw.hex()}')
    if raw[4]: raise ProtocolError(f'Servo {expected_id}: status error 0x{raw[4]:02x}')
    return raw[5:-1]

def goal_packet(goals,speeds):
    """Only 42..47 (goal position, time=0, speed); no mode/EEPROM/torque writes."""
    if not goals or set(goals)!=set(speeds): raise ValueError('Goal/speed IDs differ')
    data=bytearray([42,6])
    for sid,pos in sorted(goals.items()):
        speed=speeds[sid]
        if not 0<=sid<=253 or not 0<=pos<=4095 or not 1<=speed<=32767:
            raise ValueError('Invalid single-turn goal or bounded speed')
        data.extend(bytes([sid])+struct.pack('<HHH',pos,0,speed))
    return packet(254,0x83,data)

class Bus:
    def __init__(self,port,baud=1000000,timeout=.025):
        import serial
        self.port=serial.Serial(port=None,baudrate=baud,timeout=.002,write_timeout=.05,exclusive=True)
        self.port.dtr=False;self.port.rts=False
        self.port.port=port;self.port.open()
        self.timeout=timeout
        self.read_requests=0;self.corrupt_responses=[];self.read_failures=[]
    def close(self): self.port.close()
    def read(self,sid,address,count):
        # One fresh READ may recover a damaged/missing packet. Never use corrupt
        # data or retry a write, fault status or wrong ID. A second failure stops.
        self.read_requests+=1
        try:
            return self._read_once(sid,address,count)
        except (CorruptResponse,ReadTimeout) as exc:
            failure={'read_request':self.read_requests,'id':sid,'error':str(exc)}
            self.read_failures.append(failure)
            if isinstance(exc,CorruptResponse):self.corrupt_responses.append(failure)
            if self.read_requests>=100 and len(self.read_failures)/self.read_requests>.05:
                raise ProtocolError('Damaged/missing feedback exceeds 5% of reads') from exc
            if not getattr(self,'retry_reads',True):
                raise  # The timed controller schedules recovery after its next command.
            time.sleep(.010)
            return self._read_once(sid,address,count)
    def _read_once(self,sid,address,count):
        if not 0<=sid<=253 or not 0<=address<=255 or not 1<=count<=100: raise ValueError('Read bounds')
        command=packet(sid,2,bytes([address,count]))
        self.port.reset_input_buffer()
        if self.port.write(command)!=len(command): raise ProtocolError('Short serial write')
        deadline=time.monotonic()+self.timeout;buffer=bytearray()
        while time.monotonic()<deadline:
            buffer.extend(self.port.read(max(1,self.port.in_waiting)))
            while True:
                index=buffer.find(b'\xff\xff')
                if index<0:
                    buffer[:]=buffer[-1:] if buffer[-1:]==b'\xff' else b''
                    break
                if index: del buffer[:index]
                if len(buffer)<4: break
                size=buffer[3]+4
                if size<6 or size>106: del buffer[0];continue
                if len(buffer)<size:break
                raw=bytes(buffer[:size]);del buffer[:size]
                if raw==command:continue  # some adapters echo TX
                return decode_status(raw,sid,count)
        raise ReadTimeout(f'Servo {sid}: read timeout')
    def snapshot(self,sid):
        # Raw values retained; model-specific meaning must be verified before motion.
        config=self.read(sid,0,40);state=self.read(sid,40,31)
        u16=lambda b,i:int.from_bytes(b[i:i+2],'little')
        return {'id':sid,'model_raw':u16(config,3),'mode':config[33],
          'configured_min':u16(config,9),'configured_max':u16(config,11),
          'configured_max_temperature':config[13],'configured_max_voltage_raw':config[14],
          'configured_min_voltage_raw':config[15], 'torque_enable':state[0],
          'goal_position_raw':u16(state,2),'position_raw':u16(state,16),
          'speed_raw':u16(state,18),'load_raw':u16(state,20),
          'voltage_raw':state[22],'temperature_c':state[23],
          'status_register':state[25], 'moving':state[26], 'current_raw':u16(state,29),
          'config_0_39_hex':config.hex(),'state_40_70_hex':state.hex()}
    def feedback(self,sid):
        d=self.read(sid,56,15)
        return {'position':int.from_bytes(d[:2],'little'),'load_raw':int.from_bytes(d[4:6],'little'),
                'voltage_raw':d[6],'temperature_c':d[7],'status_register':d[9]}
    def goals(self,goals,speeds):
        goal_packet(goals,speeds)  # validate every target before the first write
        ids=sorted(goals)
        # Smaller packets bound the receive burst on the mixed-model bus.
        for offset in range(0,len(ids),10):
            group=ids[offset:offset+10]
            raw=goal_packet({sid:goals[sid] for sid in group},{sid:speeds[sid] for sid in group})
            if self.port.write(raw)!=len(raw):raise ProtocolError('Short goal write')
            self.port.flush()
            time.sleep(len(raw)*10/self.port.baudrate+.002)
