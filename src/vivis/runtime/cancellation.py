"""Recognize operator cancellation without treating hardware faults as stops."""


def operator_cancelled(exc):
    return isinstance(exc,RuntimeError) and (
        str(exc)=='Operator stop' or str(exc).startswith('Operator stop during ')
        or str(exc).startswith('Stopped during ') or str(exc)=='Stopped before audio')
