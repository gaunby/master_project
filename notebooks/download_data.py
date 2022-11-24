
from datasets import load_dataset

wiki = load_dataset("wikipedia", "20220301.en")

# save datasets
wiki.save_to_disk('/work3/s174498/concept_random_dataset/wikipedia_20220301/wiki')