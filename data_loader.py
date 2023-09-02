import json
from datasets import load_dataset, DatasetDict, Audio, Dataset, concatenate_datasets
import sys
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from whisper_normalizer.english import EnglishTextNormalizer
import gc
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="english", task="transcribe")
normalizer = EnglishTextNormalizer()
def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    #remove audio as it's not needed anymore. This saves some memory
    # encode target text to label ids 
    batch["sentence"] = normalizer(batch["sentence"])
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    batch["audio_path"] = batch["audio"]["path"]
    return batch

def create_dataset(split):
    split_file = open(f"./data/{split}.json", "r")
    split_dict = {"audio": [], "sentence": [], "dataset": []}
    for i,line in enumerate(split_file):
        line = json.loads(line)
        split_dict["audio"].append(line["audio_path"])
        split_dict["sentence"].append(line["text"])
        split_dict["dataset"].append(line["dataset"])
        
    splitset = Dataset.from_dict(split_dict).cast_column("audio", Audio())
    del split_dict
    print("Dataset loaded")
    if split == "train":
        dataset_chunks = []
        for i in range(3,4):
            gc.collect()
            chunk = splitset.shard(num_shards=4, index=i)
            chunk = chunk.map(prepare_dataset, load_from_cache_file=True, remove_columns=["audio"], num_proc=24, writer_batch_size=1, batched=False, batch_size=10, desc="Chunk {}".format(i))
            #save the processed chunks to disk
            chunk.save_to_disk(f"./data/{split}_chunk_{i}")
            dataset_chunks.append(chunk)
            gc.collect()
        return dataset_chunks
        # custom_dataset = concatenate_datasets(dataset_chunks)
        # custom_dataset.save_to_disk(f"./data/{split}")
    else:
        custom_dataset = splitset.map(prepare_dataset, load_from_cache_file=True, remove_columns=["audio"], num_proc=24, writer_batch_size=1, batched=False, batch_size=10)
        custom_dataset.save_to_disk(f"./data/{split}")
        return custom_dataset

if __name__ == "__main__":
    split = sys.argv[1]
    ds = create_dataset(split)
    print(f"Dataset {split} prepared")
    print(ds)
