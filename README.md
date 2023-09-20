# Kid-Whisper

Training code for Kid-Whisper, an adaptation of Whisper for children speech. [View paper](https://arxiv.org/pdf/2309.07927v2.pdf)
## Abstract

Recent advancements in Automatic Speech Recognition (ASR) systems, exemplified by Whisper, have demonstrated the potential of these systems to approach human-level performance given sufficient data. However, this progress doesn’t readily extend to ASR for children due to the limited availability of suitable child-specific databases and the distinct characteristics of children’s speech. A recent study investigated leveraging the My Science Tutor (MyST) children’s speech corpus to enhance Whisper’s performance in recognizing children’s speech. They were able to demonstrate some improvement on a limited testset. This paper builds on these findings by enhancing the utility of the MyST dataset through more efficient data preprocessing. We reduce the Word Error Rate (WER) on the MyST testset 13.93% to 9.11% with Whisper-Small and from 13.23% to 8.61% with Whisper-Medium and show that this improvement can be generalized to unseen datasets. We also highlight important challenges towards improving children’s ASR performance. The results showcase the viable and efficient integration of Whisper for effective children’s speech recognition.

## Contributors and authors

Ahmed Adel Attia, Jing Liu, Wei Ai, Dorottya Demszky, Carol Espy-Wilson

## Model Checkpoints
You can find the model checkpoints on my [Huggingface account](https://huggingface.co/aadel4/)

## Acknowledgments

Inspiration, code snippets, etc.
* Training script adapted from [Huggingface tutorial](https://huggingface.co/blog/fine-tune-whisper)
* Transcription and evaluation script adapted from [Open-AI code snippet](https://github.com/openai/whisper/discussions/654)
