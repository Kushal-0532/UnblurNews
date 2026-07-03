"""Simulated readers with topic preferences, producing weighted read sequences."""
import random

TOPIC_WEIGHTS = {
    "mixed": {"world": .25, "sports": .25, "business": .25, "sci_tech": .25},
    "sports_fan": {"world": .1, "sports": .7, "business": .1, "sci_tech": .1},
    "business_reader": {"world": .1, "sports": .1, "business": .7, "sci_tech": .1},
}


class SimulatedUser:
    def __init__(self, rng: random.Random, topic_weights: dict[str, float], session_length: int):
        self.rng = rng
        self.topic_weights = topic_weights
        self.session_length = session_length

    def generate_session(self, grouped_articles: dict[str, list[dict]]) -> list[dict]:
        topics = list(self.topic_weights.keys())
        weights = list(self.topic_weights.values())
        session = []
        for _ in range(self.session_length):
            topic = self.rng.choices(topics, weights=weights, k=1)[0]
            session.append(self.rng.choice(grouped_articles[topic]))
        return session


def make_users(
    n: int, scenario: str, seed: int, session_length_range: tuple[int, int] = (3, 10)
) -> list[SimulatedUser]:
    topic_weights = TOPIC_WEIGHTS[scenario]
    users = []
    for i in range(n):
        rng = random.Random(seed + i)
        session_length = rng.randint(*session_length_range)
        users.append(SimulatedUser(rng, topic_weights, session_length))
    return users
