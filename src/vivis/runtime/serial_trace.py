"""Bounded raw serial capture without changing SDK framing or timing policy."""
import json
import time
from pathlib import Path


def uart_counters(port):
    """Linux UART counters, if supplied by this serial driver."""
    import fcntl
    import struct
    try:
        values = struct.unpack('20i', fcntl.ioctl(port.fileno(), 0x545d, bytes(80)))
        return dict(zip(('cts','dsr','rng','dcd','rx','tx','frame','overrun',
                         'parity','brk','buf_overrun'), values))
    except (OSError, TypeError, ValueError, AttributeError):
        return None


class SerialTrace:
    def __init__(self, serial_port, path, max_events=100000):
        self.serial_port = serial_port
        self.path = Path(path)
        self.events = []
        self.max_events = max_events
        self.dropped_events = 0
        self.start = time.monotonic_ns()
        self.uart_before = uart_counters(serial_port)

    def __getattr__(self, name):
        return getattr(self.serial_port, name)

    def record(self, direction, **fields):
        if len(self.events) >= self.max_events:
            self.dropped_events += 1
            return
        self.events.append(dict(t_ns=time.monotonic_ns()-self.start,
                                direction=direction, **fields))

    def read(self, size=1):
        data = self.serial_port.read(size)
        if data:
            self.record('rx', hex=bytes(data).hex())
        return data

    def write(self, data):
        result = self.serial_port.write(data)
        self.record('tx', hex=bytes(data).hex(), written=result)
        return result

    def reset_input_buffer(self):
        # Observe only: do not drain/read bytes that the SDK would discard.
        self.record('clear_rx', pending_bytes=self.serial_port.in_waiting)
        return self.serial_port.reset_input_buffer()

    def close(self):
        uart_after = uart_counters(self.serial_port)
        uart_delta = ({k:uart_after[k]-v for k,v in self.uart_before.items()}
                      if self.uart_before is not None and uart_after is not None else None)
        try:
            self.serial_port.close()
        finally:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(dict(
                clock='monotonic nanoseconds since trace start',
                capture='application serial bytes, not an electrical bus capture',
                uart_before=self.uart_before, uart_after=uart_after, uart_delta=uart_delta,
                dropped_events=self.dropped_events, events=self.events), indent=2)+'\n')
