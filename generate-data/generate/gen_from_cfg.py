from nltk.grammar import CFG, Nonterminal
import sys
import itertools


def generate(filename, start=None, depth=None, n=None):
    """
    Generates an iterator of all sentences from a CFG.

    :param filename: path to file containing grammar.
    :param start: The Nonterminal from which to start generate sentences.
    :param depth: The maximal depth of the generated tree.
    :param n: The maximum number of sentences to return.
    :return: An iterator of lists of terminal tokens.
    """
    grammar = CFG.fromstring(_read_grammar(filename))
    if not start:
        start = grammar.start()
    if depth is None:
        depth = sys.maxsize

    iter = _generate_all(grammar, [start], depth)

    if n:
        iter = itertools.islice(iter, n)

    return [' '.join(string) for string in list(iter)]


def _generate_all(grammar, items, depth):
    if items:
        try:
            for frag1 in _generate_one(grammar, items[0], depth):
                for frag2 in _generate_all(grammar, items[1:], depth):
                    yield frag1 + frag2
        except RuntimeError as _error:
            if _error.message == "maximum recursion depth exceeded":
                # Helpful error message while still showing the recursion stack.
                raise RuntimeError(
                    "The grammar has rule(s) that yield infinite recursion!!"
                )
            else:
                raise
    else:
        yield []


def _generate_one(grammar, item, depth):
    if depth > 0:
        if isinstance(item, Nonterminal):
            for prod in grammar.productions(lhs=item):
                for frag in _generate_all(grammar, prod.rhs(), depth - 1):
                    yield frag
        else:
            yield [item]


def _read_grammar(filename):
    """
    Reads file and converts to string.
    :param filename: (str) ends in .txt, path to grammar
    :return: (str) grammar string as in https://www.nltk.org/_modules/nltk/parse/generate.html
    """
    with open(filename, 'r') as file:
        data = file.read()

    return data


if __name__ == "__main__":
    for n, sent in enumerate(generate('../cfgs/cfg-french.txt'), 1):
        print("%3d. %s" % (n, " ".join(sent)))