import random
from actions import ACTIONS


def no_rewrite(query):
    return query


def random_rewrite(query):
    return random.choice(ACTIONS)(query)


def static_rewrite(query):
    return query + " clinical guideline treatment"
