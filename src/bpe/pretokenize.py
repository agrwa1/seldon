import regex as re
from typing import BinaryIO
import os
from multiprocessing import Pool, cpu_count
from collections import Counter


def pretokenize_chunk(task) -> Counter:
    filepath, start, end, special_tokens, chunk_id = task

    # Split on special tokens so cannot tokenize across datasets
    # Read normally not as binary
    with open(filepath, "rb") as f:
        f.seek(start)
        chunk = f.read((end - start))
        pattern = re.compile(b"|".join([re.escape(tok) for tok in special_tokens]))
        split_chunks = re.split(pattern, chunk)

        # Pretokenize chunks
        pretokenize_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pretokenized_dict = Counter()
        for chunk in split_chunks:
            split = re.finditer(
                pretokenize_pattern, chunk.decode("utf-8", errors="ignore")
            )

            for tok in split:
                tok = tok.group().encode("utf-8", errors="ignore")
                tuple_form = tuple(bytes([b]) for b in tok)
                if tuple_form in pretokenized_dict:
                    pretokenized_dict[tuple_form] += 1
                else:
                    pretokenized_dict[tuple_form] = 1

        return pretokenized_dict


def pretokenize(filepath: str, special_tokens_input: list[str]) -> Counter:
    # Encode to binary so always workign with binary in multiprocessing function
    special_tokens = [i.encode("utf-8", errors="ignore") for i in special_tokens_input]

    with open(filepath, "rb") as f:
        num_processes = 5
        special_token = b"<|endoftext|>"
        chunk_boundaries = find_chunk_boundaries(f, special_token, num_processes)

        # Define tasks
        tasks = []
        # for chunk_id in range(1, 100):
        for chunk_id in range(1, len(chunk_boundaries)):
            start = chunk_boundaries[chunk_id - 1]
            end = chunk_boundaries[chunk_id]
            tasks.append((filepath, start, end, special_tokens, chunk_id))

        # Start parallel processing
        with Pool(processes=cpu_count()) as pool:
            partials = pool.map(pretokenize_chunk, tasks)

        # Sum together into single dict
        tuple_pretokenized = sum(partials, Counter())
        return tuple_pretokenized


def find_chunk_boundaries(
    file: BinaryIO, split_special_token: bytes, desired_num_chunks: int
) -> list[int]:
    assert isinstance(split_special_token, bytes), (
        "split_special_token must be valid bytestring"
    )

    # Get file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0, os.SEEK_SET)

    chunk_size = file_size // desired_num_chunks

    chunk_end_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_end_boundaries[-1] = file_size  # set last chunk boundary to eof

    mini_chunk_size = 4096

    for i in range(1, len(chunk_end_boundaries) - 1):
        initial_position = chunk_end_boundaries[i]
        file.seek(initial_position)
        while True:
            chunk = file.read(mini_chunk_size)

            # At EOF
            if chunk == "":
                chunk_end_boundaries[i] = file_size
                break

            found_at_index = chunk.find(split_special_token)
            if found_at_index != -1:
                # Add amount to starting index
                chunk_end_boundaries[i] = initial_position + found_at_index
                break

            initial_position += mini_chunk_size

    return sorted(set(chunk_end_boundaries))
