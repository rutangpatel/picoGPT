from datasets import load_dataset

def import_stories(train = True, test = False):
    ds = load_dataset("roneneldan/TinyStories")
    if train:
        train_ds = ds["train"]
        return train_ds
    if test:
        test_ds = ds["test"]
        return test_ds