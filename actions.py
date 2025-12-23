def add_domain_terms(query):
    return query + " type 2 metformin first line treatment"

def add_guideline_terms(query):
    return query + " clinical guideline recommended therapy"

def expand_acronyms(query):
    return query.replace("dm", "diabetes mellitus")

def no_op(query):
    return query


ACTIONS = [
    add_domain_terms,
    add_guideline_terms,
    expand_acronyms,
    no_op
]
