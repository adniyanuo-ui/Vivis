"""Bounded public invitations. Their text never contains a private record."""
from math import isfinite


def public_invitation(address=None, previous_turns=(), *, welcome=False, index=0):
    from vivis.speech.fixed_replies import REPLIES
    questions = {
        '表情跟随': '想试试表情跟随吗？',
        '剧本': '想看看有哪些剧本吗？',
        '歌曲': '想看看有哪些歌曲吗？',
    }
    fallback = tuple(REPLIES[key] for key in ('idle_first','idle_second','idle_third'))
    question = fallback[index % len(fallback)]
    if welcome:
        # Match a public category only. Never repeat a historical quotation.
        for turn in previous_turns:
            if turn.get('role') != 'user':
                continue
            topic = next((name for name in questions if name in turn.get('text', '')), None)
            if topic:
                question = questions[topic]
                break
        if address:return address + '，您好！' + question
        return REPLIES['public_welcome']
    return question


class ReceptionConversation:
    def __init__(self, config):
        self.enabled = config.get('enabled', False)
        self.delay = config.get('first_prompt_s', 6)
        self.interval = config.get('min_interval_s', 12)
        self.limit = config.get('max_per_visit', 6)
        if (type(self.enabled) is not bool or type(self.limit) is not int or not 1 <= self.limit <= 6
                or any(type(n) not in (int, float) or not isfinite(n) for n in (self.delay, self.interval))
                or not 6 <= self.delay <= 60 or not 12 <= self.interval <= 600):
            raise ValueError('Invalid reception conversation configuration')
        self.reset()

    def reset(self):
        self.quiet_since = None
        self.last_prompt = None
        self.last_attempt = None
        self.count = 0
        self.paused = False
        self.declined = False
        self.topics = []

    def user_started(self):
        self.quiet_since = None

    def defer(self, kind):
        if kind == 'thinking':
            self.paused = True
        elif kind == 'declined':
            self.declined = True
        else:
            raise ValueError('Invalid reception preference')

    def addressed_response(self):
        self.paused = False

    def due(self, now, *, available):
        if not available:
            self.quiet_since = None
            return False
        if self.quiet_since is None:
            self.quiet_since = now
        return (self.enabled and not self.paused and not self.declined
                and self.count < self.limit and now - self.quiet_since >= self.delay
                and (self.last_prompt is None or now - self.last_prompt >= self.interval)
                and (self.last_attempt is None or now - self.last_attempt >= self.interval))

    def played(self, now):
        self.last_prompt = now
        self.count += 1
        self.quiet_since = None

    def remember_topic(self,topic):
        if topic not in self.topics:self.topics.append(topic)
        self.topics=self.topics[-self.limit:]
