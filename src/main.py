from bpe import train_bpe
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main():
    training_set_filepath = "./data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 300
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(training_set_filepath, vocab_size, special_tokens)


if __name__ == "__main__":
    logging.info("Starting main function")

    main()
