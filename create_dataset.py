import json
from datasets import load_dataset, DatasetDict, Audio, Dataset, concatenate_datasets
import sys
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from whisper_normalizer.english import EnglishTextNormalizer
from collections import defaultdict
import gc
import os
def create_dataset(split, english = True):
    base_model = "openai/whisper-small"
    outdir = f"./data/"
    if english == True:
        base_model += ".en"
    else:
        outdir += "multilingual/"
    outdir += f"{split}"
    print(f"Output directory: {outdir}")
    print(f"Base model: {base_model}")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(base_model)
    tokenizer = WhisperTokenizer.from_pretrained(base_model, language="english", task="transcribe")
    normalizer = EnglishTextNormalizer()
    
    os.makedirs(outdir, exist_ok=True)
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

    split_file = open(f"./data/{split}.json", "r")
    split_dict = defaultdict(lambda: {"audio": [], "sentence": [], "dataset": []})
    for i,line in enumerate(split_file):
        line = json.loads(line)
        dataset = line["dataset"]
        
        if dataset == "cslu":
            if "spontaneous" in line["audio_path"]:
                dataset = "cslu_spontaneous"
            else:
                dataset = "cslu_scripted"
        # else:
            # continue
        split_dict[dataset]["audio"].append(line["audio_path"])
        split_dict[dataset]["sentence"].append(line["text"])
        split_dict[dataset]["dataset"].append(dataset)
    print(split_dict.keys())
    datasets = {ds: Dataset.from_dict(split_dict[ds]).cast_column("audio", Audio()) for ds in split_dict}
    del split_dict
    print("Dataset loaded")
        # custom_dataset = splitset.map(prepare_dataset, load_from_cache_file=True, remove_columns=["audio"], num_proc=24, writer_batch_size=1, batched=False, batch_size=10)
    # custom_dataset.save_to_disk(f"{outdir}")
    custom_dataset = DatasetDict()
    for dataset_name, dataset in datasets.items():
        if split == "train" and dataset_name == "myst":
            # continue #skip myst for now
            for i in range(0,1):
                gc.collect()
                chunk = dataset.shard(num_shards=4, index=i)
                chunk = chunk.map(prepare_dataset, load_from_cache_file=True, remove_columns=["audio"], num_proc=24, writer_batch_size=1, batched=False, batch_size=10, desc="Chunk {}".format(i))
                #save the processed chunks to disk
                chunk.save_to_disk(f"{outdir}/{dataset_name}/chunk_{i}")
                print(f"Chunk {i} processed")
                gc.collect()
        # custom_dataset = concatenate_datasets(dataset_chunks)
        # custom_dataset.save_to_disk(f"{outdir}")
        else:
            print(dataset_name, dataset)
            dataset = dataset.map(prepare_dataset, load_from_cache_file=True, remove_columns=["audio"], num_proc=24, writer_batch_size=1, batched=False, batch_size=10)
            dataset.save_to_disk(f"{outdir}/{dataset_name}")
            custom_dataset[dataset_name] = dataset
    return custom_dataset

if __name__ == "__main__":
    split = sys.argv[1]
    english = sys.argv[2] if len(sys.argv) > 2 else True
    #cast english from str to bool
    english = english == "True"
    print(f"Preparing dataset {split}")
    print(f"English: {english}")
    ds = create_dataset(split, english = english)
    
    print(f"Dataset {split} prepared")
    print(ds)
