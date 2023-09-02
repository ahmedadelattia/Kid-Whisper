from transformers import pipeline, WhisperForConditionalGeneration
from transformers import WhisperFeatureExtractor
from collections import defaultdict
from datasets import Dataset, load_dataset, Audio
import soundfile as sf
from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
from evaluate import load

def load_data():
    #load chunks from disk
    custom_dataset = Dataset.load_from_disk("./data/test")
    print("Dataset prepared")
    print(f"Found {len(custom_dataset)} test examples")
    assert "input_features" in custom_dataset.column_names, "input_features not in custom_dataset"
    assert "labels" in custom_dataset.column_names, "labels not in custom_dataset"
    
    return custom_dataset


def transcribe(model_name):
    testset = load_data()
    testset = testset.remove_columns(["input_features"])
    testset = testset.cast_column("audio_path", Audio())
    testset = testset.rename_column("audio_path", "audio")
    # base_model = "openai/" + model_name.split("openai/")[1].split("/")[0]
    base_model = "openai/whisper-small"
    processor = WhisperProcessor.from_pretrained(base_model)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to("cuda")

    def map_to_pred(batch):
        audio = batch["audio"]
        input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
        scentence = processor.decode(batch["labels"])
        batch["reference"] = processor.tokenizer._normalize(scentence)
        with torch.no_grad():
            predicted_ids = model.generate(input_features.to("cuda"))[0]
        transcription = processor.decode(predicted_ids)
        batch["prediction"] = processor.tokenizer._normalize(transcription)
        batch["audio_path"] = batch["audio"]["path"]
        batch["dataset"] = batch["dataset"]
        return batch

    result = testset.map(map_to_pred)

    metric = load("wer")
    datasets = defaultdict(lambda: {"ground_truths": [], "hypotheses": []})
    wer = defaultdict(lambda: 0)
    for i,batch in enumerate(result):
        ds = batch["dataset"]
        datasets[ds]["hypotheses"].append(batch["prediction"])
        datasets[ds]["ground_truths"].append(batch["reference"])
        
    for ds in datasets:
        wer[ds] = 100*metric.compute(predictions=datasets[ds]["hypotheses"], references=datasets[ds]["ground_truths"])
        print(f"Dataset: {ds} WER: {wer[ds]}")
    
    print(f"{wer}")
    with open(f"./{model_name}/transcriptions.txt", "w") as f:
        for i, batch in enumerate(result):
            transcription = batch["prediction"]
            line = batch["audio_path"] + "\t" + transcription
            f.write(f"{line}\n")
    with open(f"./{model_name}/wer.txt", "w") as f:
        f.write(f"{wer}")
        
        
transcribe("whisper-small/checkpoint-3000")