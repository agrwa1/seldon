from .pretokenize import pretokenize


def bpe(filepath: str):
    pretokenized = pretokenize(filepath)
    print(pretokenized.most_common(20))
