"""Official FTServo_Python transport, retaining the Xiaobai motion/feedback guards.

Pinned source: xiaobai/reference/FTServo_Python (MIT, see its LICENSE).
Only RAM 41..47 is written; acceleration is preserved from the initial readback.
"""
import sys
import struct
import time
from pathlib import Path
from .protocol import Bus, ProtocolError, CorruptResponse, ReadTimeout, ServoStatusError, goal_packet

SDK_ROOT = Path(__file__).resolve().parents[2] / 'xiaobai/reference/FTServo_Python'
sys.path.insert(0, str(SDK_ROOT))
from scservo_sdk import PortHandler, sms_sts, GroupSyncWrite, COMM_SUCCESS, COMM_RX_TIMEOUT, COMM_RX_CORRUPT


class OfficialSDKBus(Bus):
    def __init__(self, port, baud=1000000, feedback_mode='block',write_mode='position_speed',changed_goals_only=False,trace_path=None,post_goal_quiet_ms=0,sync_group_size=30,sync_group_gap_ms=0,transaction_timeout_ms=None,servo_status_policy='raise'):
        if servo_status_policy not in ('raise','record_only'):
            raise ValueError('Unknown servo status policy')
        self.servo_status_policy=servo_status_policy
        self.servo_status_notifications=[]
        if transaction_timeout_ms is not None:
            if type(transaction_timeout_ms) not in (int,float) or not 2<=transaction_timeout_ms<=50:
                raise ValueError('Transaction timeout must be 2..50 ms')
        if type(sync_group_size) is not int or not 1<=sync_group_size<=30 or not 0<=sync_group_gap_ms<=2:
            raise ValueError('Invalid sync group size or gap')
        self.sync_group_size=sync_group_size
        self.sync_group_gap_ms=sync_group_gap_ms
        if not 0 <= post_goal_quiet_ms <= 5:
            raise ValueError('Post-goal quiet interval must be 0..5 ms')
        self.post_goal_quiet_ms = post_goal_quiet_ms
        self.baud = baud
        if feedback_mode not in ['block','short_registers']:
            raise ValueError('Unknown SDK feedback mode')
        self.feedback_mode = feedback_mode
        if write_mode not in ['position_speed','position_only']:raise ValueError('Unknown SDK write mode')
        self.write_mode=write_mode
        self.handler = PortHandler(port)
        if not self.handler.setBaudRate(baud):
            raise ProtocolError('Official SDK could not open serial port')
        self.port = self.handler.ser
        self.port.exclusive = True
        self.transaction_timeout_ms=transaction_timeout_ms
        if transaction_timeout_ms is not None:
            # Instance-local deadline: retain the pinned SDK and its global
            # defaults. A timeout remains an error; no missing data is accepted.
            self.handler.setPacketTimeout=lambda length:self.handler.setPacketTimeoutMillis(transaction_timeout_ms)
        if trace_path is not None:
            from .serial_trace import SerialTrace
            self.handler.ser = SerialTrace(self.port,trace_path)
        self.sdk = sms_sts(self.handler)
        self.position_writer=GroupSyncWrite(self.sdk,42,2)
        self.read_requests = 0
        self.corrupt_responses = []
        self.read_failures = []
        self.acceleration = {}
        self.configured_speeds = {}
        self.goal_packets_sent = 0
        self.last_sdk_response = None
        self.changed_goals_only = changed_goals_only
        self.last_sent_commands = {}
        self.goal_axes_sent = 0
        self.goal_axes_suppressed = 0
        self.last_transmitted_ids = []

    def close(self):
        self.handler.closePort()

    def _read_once(self, sid, address, count):
        if not 0 <= sid <= 253 or not 0 <= address <= 255 or not 1 <= count <= 100:
            raise ValueError('Read bounds')
        data, result, error = self.sdk.readTxRx(sid, address, count)
        self.last_sdk_response = {'id':sid,'address':address,'requested_bytes':count,
                                  'communication_result':result,'servo_status':error,'data_hex':bytes(data).hex()}
        if result == COMM_RX_TIMEOUT:
            raise ReadTimeout(f'Official SDK ID {sid}: read timeout')
        if result == COMM_RX_CORRUPT:
            raise CorruptResponse(f'Official SDK ID {sid}: corrupt response')
        if result != COMM_SUCCESS:
            raise ProtocolError(f'Official SDK ID {sid}: {self.sdk.getTxRxResult(result)}')
        if len(data) != count:
            raise CorruptResponse(f'Official SDK ID {sid}: unexpected payload length')
        if error:
            if getattr(self,'servo_status_policy','raise')!='record_only':
                raise ServoStatusError(sid,error)
            # The SDK has verified checksum/ID/length and COMM_SUCCESS above.
            # An explicit observation policy records alarm bits without throwing
            # away that valid payload. Damaged or missing packets still fail.
            self.servo_status_notifications.append({**self.last_sdk_response,
                                                    'monotonic':time.monotonic()})
        return bytes(data)

    def snapshot(self, sid):
        state = super().snapshot(sid)
        self.acceleration[sid] = bytes.fromhex(state['state_40_70_hex'])[1]
        self.configured_speeds[sid] = int.from_bytes(bytes.fromhex(state['state_40_70_hex'])[6:8],'little')
        return state

    def feedback(self, sid):
        if self.feedback_mode == 'block':
            return super().feedback(sid)
        # Keep all feedback checks, but compare short register reads with a
        # 15-byte live RAM read while the servo updates its feedback registers.
        position = int.from_bytes(self.read(sid,56,2),'little')
        load = int.from_bytes(self.read(sid,60,2),'little')
        sensors = self.read(sid,62,2)
        status = self.read(sid,65,1)[0]
        return {'position':position,'load_raw':load,'voltage_raw':sensors[0],
                'temperature_c':sensors[1],'status_register':status}

    def goals(self, goals, speeds):
        goal_packet(goals, speeds)  # validate all positions/speeds before any write
        if any(sid not in self.acceleration for sid in goals):
            raise ProtocolError('Need initial snapshot to preserve acceleration')
        self.last_transmitted_ids = []
        commands = {sid:(goals[sid],speeds[sid],self.acceleration[sid]) for sid in goals}
        if getattr(self,'changed_goals_only',False):
            # Windows ServoServer suppresses unchanged integer goals. Include
            # speed/acceleration here so configuration changes are not dropped.
            previous = getattr(self,'last_sent_commands',{})
            changed = {sid:pos for sid,pos in goals.items() if previous.get(sid)!=commands[sid]}
            self.goal_axes_suppressed += len(goals)-len(changed)
            goals = changed
        if not goals:
            return
        group_size=getattr(self,'sync_group_size',30)
        ordered=sorted(goals)
        for offset in range(0,len(ordered),group_size):
            group={sid:goals[sid] for sid in ordered[offset:offset+group_size]}
            self._send_goals(group,speeds)
            if getattr(self,'sync_group_gap_ms',0):
                self.port.flush()
                width=3 if self.write_mode=='position_only' else 8
                time.sleep((8+width*len(group))*10/self.baud+self.sync_group_gap_ms/1000)
        if getattr(self,'post_goal_quiet_ms',0):
            # tcdrain can finish at the USB driver boundary. Also budget a full
            # 8N1 packet on the UART plus a quiet interval before the next read.
            self.port.flush()
            width = 3 if self.write_mode == 'position_only' else 8
            time.sleep((8+width*len(goals))*10/self.baud+self.post_goal_quiet_ms/1000)
        # Commit only after successful transmission; a failed write cannot make
        # the next attempt incorrectly suppress a command that was never sent.
        if not hasattr(self,'last_sent_commands'):self.last_sent_commands={}
        self.last_sent_commands.update({sid:commands[sid] for sid in goals})
        self.last_transmitted_ids = sorted(goals)
        self.goal_axes_sent = getattr(self,'goal_axes_sent',0)+len(goals)

    def _send_goals(self, goals, speeds):
        if getattr(self,'write_mode','position_speed')=='position_only':
            if any(self.configured_speeds[sid]!=speeds[sid] for sid in goals):
                raise ProtocolError('Position-only test requires existing bounded device speeds to match profile')
            self.position_writer.clearParam()
            try:
                for sid in sorted(goals):
                    if not self.position_writer.addParam(sid,list(struct.pack('<H',goals[sid]))):
                        raise ProtocolError('Cannot enqueue position')
                result=self.position_writer.txPacket()
                if result!=COMM_SUCCESS:raise ProtocolError(self.sdk.getTxRxResult(result))
                self.goal_packets_sent+=1
            finally:self.position_writer.clearParam()
            return
        # Official SDK limit: 250 bytes; 29 motors * 8 + 8 = 240 bytes.
        if len(goals) > 30:
            raise ValueError('Too many motors for one official SyncWritePosEx packet')
        self.sdk.groupSyncWrite.clearParam()
        try:
            for sid in sorted(goals):
                if not self.sdk.SyncWritePosEx(sid, goals[sid], speeds[sid], self.acceleration[sid]):
                    raise ProtocolError('Official SDK failed to enqueue target')
            result = self.sdk.groupSyncWrite.txPacket()
            if result != COMM_SUCCESS:
                raise ProtocolError(self.sdk.getTxRxResult(result))
            self.goal_packets_sent += 1
        finally:
            self.sdk.groupSyncWrite.clearParam()
