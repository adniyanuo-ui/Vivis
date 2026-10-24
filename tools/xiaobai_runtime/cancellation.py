"""Distinguish an operator stop from transport and actuator faults."""


class OperatorStop(RuntimeError):
    """Stop requested by a signal or operator stop file."""
