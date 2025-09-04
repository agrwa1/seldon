from bpe import bpe


def main():
    training_set_filepath = "./data/TinyStoriesV2-GPT4-valid.txt"
    bpe(training_set_filepath)


if __name__ == "__main__":
    main()
