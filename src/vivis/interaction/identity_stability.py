"""Fast acquisition of verified identity, delayed replacement/loss for reception."""
import threading


class IdentityStability:
    def __init__(self, hold_s):
        if not isinstance(hold_s, (int, float)) or not 1 <= hold_s <= 60:
            raise ValueError('Invalid identity hold duration')
        self.hold_s = hold_s
        self.person = None
        self.change_started = None
        self.released_since = None
        self.last_now = None
        self.lock = threading.RLock()

    def observe(self, verified_person, now):
        """Input must already pass recognition confidence and sample confirmation.

        An unchanged identity cancels a pending change. Different candidates and
        missing faces cannot extend the original eight-second deadline. Once
        released, unknown input stays unknown; it cannot resurrect the old person.
        """
        with self.lock:
            if self.last_now is not None and now < self.last_now:
                self.person = None
                self.change_started = None
                self.released_since = None
            self.last_now = now
            if self.person is None:
                if verified_person is not None:
                    self.person = verified_person
                    self.change_started = None
                    self.released_since = None
            elif verified_person == self.person:
                self.change_started = None
                self.released_since = None
            else:
                if self.change_started is None:
                    self.change_started = now
                if now - self.change_started >= self.hold_s:
                    self.person = verified_person
                    self.released_since = self.change_started if verified_person is None else None
                    self.change_started = None
            holding = self.person is not None and self.change_started is not None
            return dict(person_id=self.person,
                        state='reviewing' if holding else 'recognized' if self.person else 'unknown',
                        held=holding, candidate_id=verified_person,
                        change_started=self.change_started, released_since=self.released_since,
                        remaining_s=max(0., self.hold_s-(now-self.change_started)) if holding else 0.)
