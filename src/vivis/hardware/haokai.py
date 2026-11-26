"""HK-QXMIC V3 direction frames. No microphone, actuator or board writes."""
from dataclasses import dataclass
import math
from pathlib import Path


HEADER = b'\x55\xaa'


@dataclass(frozen=True)
class DirectionFrame:
    sector: int
    raw: bytes


class DirectionParser:
    """Resynchronise a byte stream; suffix bytes in the vendor example are unknown."""
    def __init__(self):
        self.pending = bytearray()
        self.bytes_received = self.frames = self.invalid = self.discarded = 0

    def feed(self, data):
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError('Direction data must be bytes')
        self.bytes_received += len(data)
        self.pending.extend(data)
        result = []
        while self.pending:
            start = self.pending.find(HEADER)
            if start < 0:
                keep = int(self.pending[-1] == HEADER[0])
                self.discarded += len(self.pending) - keep
                self.pending[:] = self.pending[-1:] if keep else b''
                break
            if start:
                self.discarded += start
                del self.pending[:start]
            if len(self.pending) < 4:
                break
            sector, checksum = self.pending[2:4]
            if sector < 12 and checksum == 0xff - sector:
                result.append(DirectionFrame(sector, bytes(self.pending[:4])))
                self.frames += 1
                del self.pending[:4]
            else:
                self.invalid += 1
                self.discarded += 1
                del self.pending[0]
        return result

    def statistics(self):
        return dict(bytes_received=self.bytes_received, valid_frames=self.frames,
                    invalid_frames=self.invalid, discarded_bytes=self.discarded,
                    pending_bytes=len(self.pending))


class DirectionState:
    """Host receive time is not a speech timestamp or a voice-activity indicator."""
    def __init__(self, coordinates, max_age_s):
        self.coordinates = coordinates
        self.max_age_s = max_age_s
        self.frame = None
        self.received = None
        self.sequence = 0

    def observe(self, frame, now):
        if (not isinstance(frame, DirectionFrame) or not 0 <= frame.sector < 12
                or frame.raw != HEADER + bytes((frame.sector, 0xff-frame.sector))
                or not math.isfinite(now)):
            raise ValueError('Invalid direction observation')
        self.frame, self.received = frame, now
        self.sequence += 1

    def snapshot(self, now):
        if not math.isfinite(now):
            raise ValueError('Invalid observation clock')
        age = None if self.received is None else now-self.received
        fresh = age is not None and 0 <= age <= self.max_age_s
        frame = self.frame if fresh else None
        nominal = ((self.coordinates['sector_zero_nominal_deg'] +
                    frame.sector*self.coordinates['sector_step_deg']) % 360
                   if frame else None)
        return dict(schema_version='vivis-sound-direction-observation-1.0',
                    status='fresh' if fresh else 'waiting' if age is None else 'stale',
                    sequence=self.sequence, received_monotonic=self.received,
                    age_s=age, frame_valid=bool(frame),
                    sector=frame.sector if frame else None,
                    board_nominal_azimuth_deg=nominal,
                    sector_step_deg=self.coordinates['sector_step_deg'],
                    coordinate_reference=self.coordinates['reference'],
                    raw_frame_hex=frame.raw.hex(' ').upper() if frame else None,
                    speech_active=None, confidence=None, robot_azimuth_deg=None,
                    elevation_deg=None, distance_m=None, motion_authorized=False)


def select_port(ports, settings, protected_port, requested=None):
    """Only a unique matching USB adapter; never fall back to the head's port."""
    selected = requested or settings.get('device')
    protected = Path(protected_port).resolve()
    if selected and Path(selected).resolve() == protected:
        raise ValueError('拒绝打开舵机串口；声源定位必须使用独立CH340串口')
    candidates = [p for p in ports
                  if p.vid == settings['usb_vid'] and p.pid == settings['usb_pid']
                  and Path(p.device).resolve() != protected]
    if selected:
        candidates = [p for p in candidates if Path(p.device).resolve() == Path(selected).resolve()]
    if not candidates:
        raise ValueError('未找到匹配的CH340声源串口；请先检查USB枚举和接线')
    if len(candidates) != 1:
        raise ValueError('发现多个CH340串口，请用--port指定声源定位的独立端口')
    return str(Path(candidates[0].device).resolve())


def open_receiver(port, settings):
    """Configure only the host UART. Never send a query or board command."""
    import serial
    receiver = serial.Serial(port=None, baudrate=settings['baud_rate'],
                             bytesize=settings['data_bits'], parity=settings['parity'],
                             stopbits=settings['stop_bits'], timeout=settings['read_timeout_s'],
                             write_timeout=settings['read_timeout_s'], exclusive=True,
                             xonxoff=False, rtscts=False, dsrdtr=False)
    receiver.dtr = receiver.rts = False
    receiver.port = port
    try:
        receiver.open()
    except BaseException:
        try:
            receiver.close()
        except OSError:
            pass  # Preserve the original open failure if USB teardown also fails.
        raise
    return receiver
