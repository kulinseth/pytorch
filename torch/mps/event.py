import torch

class Event:
    r"""Wrapper around a MPS event.

    MPS events are synchronization markers that can be used to monitor the
    device's progress, to accurately measure timing, and to synchronize MPS streams.

    Args:
        enable_timing (bool, optional): indicates if the event should measure time
            (default: ``False``)
    """

    def __init__(self, enable_timing=False):
        self.eventId = torch._C._mps_acquireEvent(enable_timing)

    def __del__(self):
        if self.eventId > 0:
            torch._C._mps_releaseEvent(self.eventId)

    def record(self):
        r"""Records the event in the default stream."""
        torch._C._mps_recordEvent(self.eventId)

    def wait(self):
        r"""Makes all future work submitted to the default stream wait for this event."""
        torch._C._mps_waitForEvent(self.eventId)

    def query(self):
        r"""Checks if all work currently captured by event has completed.

        Returns:
            A boolean indicating if all work currently captured by event has
            completed.
        """
        return torch._C._mps_queryEvent(self.eventId)

    def synchronize(self):
        r"""Waits for the event to complete.

        Waits until the completion of all work currently captured in this event.
        This prevents the CPU thread from proceeding until the event completes.
        """
        torch._C._mps_synchronizeEvent(self.eventId)
