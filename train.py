from datasets import load_dataset, DatasetDict, Audio, Dataset, concatenate_datasets
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration, WhisperModel
from whisper_normalizer.english import EnglishTextNormalizer
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from transformers import EarlyStoppingCallback
import json
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
import argparse
from datetime import datetime
import os

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a Whisper model for ASR")
    parser.add_argument(
        "--pretrained_model",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
        default="openai/whisper-small.en",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to dataset",
        default="./data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to output directory",
        default="./fine-tuned-whisper",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        help="Number of training epochs",
        default=10,
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        help="Training batch size",
        default=16,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        help="Gradient accumulation steps",
        default=1,
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        help="Evaluation batch size",
        default=8,
    )
    parser.add_argument(
        "--max_learning_rate",
        type=float,
        help="Maximum learning rate",
        default=1e-5,
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        help="Number of warmup steps",
        default=500,
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        help="Maximum number of training steps",
        default=4000,
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        help="Evaluation steps",
        default=1000,
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=bool,
        help="Use gradient checkpointing",
        default=True,
    )
    parser.add_argument(
        "--fp16",
        type=bool,
        help="Use fp16",
        default=True,
    )
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        help="Evaluation strategy",
        default="steps",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        help="Logging steps",
        default=500,
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        help="Save steps",
        default=1000,
    )
    print("Parser created")
    return parser
        
        


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    



def load_data():
    #load chunks from disk
    dataset_chunks = []
    for i in range(0, 4):
        chunk = Dataset.load_from_disk("./data/train_chunk_{}".format(i))
        dataset_chunks.append(chunk)
    custom_dataset = DatasetDict({"train": concatenate_datasets(dataset_chunks), "development": Dataset.load_from_disk("./data/development")})
    print("Dataset prepared")
    print("Found {} training examples and {} development examples".format(len(custom_dataset["train"]), len(custom_dataset["development"])))
    assert "input_features" in custom_dataset["train"].column_names, "input_features not in custom_dataset"
    assert "labels" in custom_dataset["train"].column_names, "labels not in custom_dataset"
    assert "input_features" in custom_dataset["development"].column_names, "input_features not in custom_dataset"
    assert "labels" in custom_dataset["development"].column_names, "labels not in custom_dataset"

    return custom_dataset

def train(args):
    
    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array 
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        #remove audio as it's not needed anymore. This saves some memory
        # encode target text to label ids 
        batch["sentence"] = normalizer(batch["sentence"])
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        
        return batch
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        pred_str = [normalizer(text) for text in pred_str]
        label_str = [normalizer(text) for text in label_str]
        # filtering step to only evaluate the samples that correspond to non-zero references
        pred_str = [pred_str[i] for i in range(len(pred_str)) if len(label_str[i]) > 0]
        label_str = [label_str[i] for i in range(len(label_str)) if len(label_str[i]) > 0]

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    
    normalizer = EnglishTextNormalizer()
    metric = evaluate.load("wer")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.pretrained_model)
    tokenizer = WhisperTokenizer.from_pretrained(args.pretrained_model, language="english", task="transcribe")
    processor = WhisperProcessor.from_pretrained(args.pretrained_model, language="english", task="transcribe")
    custom_dataset = load_data()
    model = WhisperForConditionalGeneration.from_pretrained(args.pretrained_model)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    if ".en" not in args.pretrained_model:
        model.config.forced_decoder_ids = None #remove forced decoder ids for non-english models
    # model.config.forced_decoder_ids = None
    model.config.use_cache = False
    model.config.suppress_tokens = []
    
    if args.num_train_epochs > 0:
        args.max_steps = 0
        args.evaluation_strategy = "steps"
        print("Training model for {} epochs".format(args.num_train_epochs))
        print("max steps ignored")
        
    output_dir = args.output_dir + "/" +args.pretrained_model +"/lr_{}_warmup_{}_epochs_{}_batch_{}_grad_acc_{}_max_steps_{}".format(args.max_learning_rate, args.warmup_steps, args.num_train_epochs, args.train_batch_size, args.gradient_accumulation_steps, args.max_steps) + "/{}".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(output_dir, exist_ok=True)
    #dump args to file
    with open(output_dir + "/args.json", "w") as f:
        json.dump(args.__dict__, f)
    print("Output directory: {}".format(output_dir))
    
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.max_learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        evaluation_strategy=args.evaluation_strategy,
        per_device_eval_batch_size=args.eval_batch_size,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=True,
    )
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=custom_dataset["train"],
        eval_dataset=custom_dataset["development"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    
    processor.save_pretrained(training_args.output_dir)
    trainer.train()
    
    trainer.save_model()
    
    print("Training finished")
    print("Model saved to {}".format(training_args.output_dir))
    
    return training_args.output_dir

def main():
    parser = get_parser()
    args = parser.parse_args()
    train(args)
    
if __name__ == "__main__":
    main()
    #example: python train.py --num_train_epochs 10 --train_batch_size 16 --gradient_accumulation_steps 1 --eval_batch_size 8 --max_learning_rate 1e-5 --warmup_steps 500 --max_steps 4000 --gradient_checkpointing True --fp16 True --evaluation_strategy steps --logging_steps 500 --save_steps 1000 --output_dir ./fine-tuned-whisper