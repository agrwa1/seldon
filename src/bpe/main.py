from .pretokenize import pretokenize
import logging
import time
from collections import Counter

logger = logging.getLogger(__name__)


def train_bpe(
    filepath: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Returns (vocab, merges) from BPE
    """
    logger.info("Starting pretokenization")
    pretokenization_start = time.time()
    pretokenized = pretokenize(filepath, special_tokens)
    logger.info(
        f"Finished pretokenization in {(time.time() - pretokenization_start):.4f} seconds"
    )

    logger.info("Starting BPE merges")
    bpe_merge_start = time.time()
    vocab, merges = bpe_merge(pretokenized, vocab_size, special_tokens)
    # Add special tokens to vocab
    logger.info(
        f"BPE merging finished in {(time.time() - bpe_merge_start):.4f} seconds"
    )

    print(merges)

    return ({}, [])


def bpe_merge(
    pretokenized: Counter, max_vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    print(pretokenized.most_common(5))
    # Optimize this!
    # Starting vocab size is 256 for each byte + special tokens
    vocab = {i: bytes([i]) for i in range(256)}
    for i in range(len(special_tokens)):
        vocab[len(vocab)] = special_tokens[i].encode("utf-8", errors="ignore")
    merges = []

    while len(vocab.keys()) < max_vocab_size:
        if len(vocab) % 10 == 0:
            logger.info(f"Vocab size: {len(vocab)}")
        # Slow step: create pair counts for all pretokenized
        pair_counts = Counter()
        for seq, freq in pretokenized.items():
            for j in range(len(seq) - 1):
                pair_counts[(seq[j], seq[j + 1])] += freq

        # If no more merges to make
        if not pair_counts:
            break

        best_pair = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        x, y = best_pair
        merged = x + y

        # Merge max occuring byte pair and add to vocab
        merges.append(merged)
        vocab[len(vocab.keys())] = merged

        new_pretokenized = Counter()
        for seq, freq in pretokenized.items():
            lst = list(seq)
            out = []
            i = 0
            n = len(seq)
            while i < n:
                if i + 1 < n and lst[i] == x and lst[i + 1] == y:
                    out.append(merged)
                    i += 2
                else:
                    out.append(lst[i])
                    i += 1
            new_pretokenized[tuple(out)] += freq
        pretokenized = new_pretokenized

    return (vocab, merges)
