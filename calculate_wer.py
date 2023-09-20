from evaluate import load
from datasets import Dataset, load_dataset, Audio
from collections import defaultdict
import json
# from whisper_normalizer.english import EnglishTextNormalizer
from transformers import WhisperTokenizer
import sys
from tqdm import tqdm
from Rishabh_norm import RishabhTextNormalizer
transcription_file = sys.argv[1]
transcriptions = open(transcription_file, "r").readlines()
if "librispeech" in transcription_file.lower():
    test_set = load_dataset("librispeech_asr", "clean", split="test")
    test_set = {line["audio"]["path"]: line["text"] for line in test_set}
elif "cslu_spontaneous" in transcription_file.lower():
    test_set = open("data/cslu_spontaneous.json", "r").readlines()
    test_set = [json.loads(line) for line in test_set]
    test_set = {line["audio_path"]: line for line in test_set}
else:
    test_set = open("./data/test.json", "r").readlines()
    test_set = [json.loads(line) for line in test_set]
    test_set = {line["audio_path"]: line for line in test_set}
datasets = defaultdict(lambda: {"ground_truths": [], "hypotheses": []})

# if "rishabh" in transcription_file.lower():
#     normalizer = RishabhTextNormalizer
#     print("Using Rishabh normalizer")
# else:
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small.en", language="english", task="transcribe")
normalizer = tokenizer._normalize
metric = load("wer")
print(f"Found {len(transcriptions)} transcriptions")
for i,line in tqdm(enumerate(transcriptions), desc="Processing transcriptions", total=len(transcriptions)):
    path, hyp = line.split("\t")
    if "librispeech" in transcription_file.lower():
        gt = test_set[path]
        ds = "librispeech"
    else:
        gt = test_set[path]["text"]
        ds = test_set[path]["dataset"]
    wer = metric.compute(predictions=[normalizer(hyp)], references=[normalizer(gt)]) * 100
    # if wer > 40:
    #     print(f"WER for {path} is {wer}")
    #     print(f"GT: {normalizer(gt)}")
    #     print(f"HYP: {normalizer(hyp)}")
    #     print(f"Dataset: {ds}")
    #     path = path.replace(" ", "\ ")
    #     print(f"Path: {path}")
    #     # exit()
    #     continue
    datasets[ds]["hypotheses"].append(normalizer(hyp))
    datasets[ds]["ground_truths"].append(normalizer(gt))

    
metric = load("wer")
gt,ht= [], []

for ds in datasets:
    wer = metric.compute(predictions=datasets[ds]["hypotheses"], references=datasets[ds]["ground_truths"]) * 100
    print("Dataset: {} WER: {}".format(ds, wer))
    gt.extend(datasets[ds]["ground_truths"]) 
    ht.extend(datasets[ds]["hypotheses"])   
wer = metric.compute(predictions=ht, references=gt) * 100
print("WER: {}".format(wer))