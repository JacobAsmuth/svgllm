from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Tuple
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, LlavaForConditionalGeneration, TrainingArguments, Trainer

from PIL import Image

from svgllm.data.svg_dataset import SvgSftDataset


PROMPT_TEXT = "Reproduce this image as an SVG."


@dataclass
class Collator:
    processor: AutoProcessor
    max_length: int

    def __call__(self, batch: List) -> dict:
        images: List[Image.Image] = [ex.image for ex in batch]
        svg_texts: List[str] = [ex.svg_text for ex in batch]

        # Build per-sample chat texts: prompt-only and full (with assistant)
        prompt_texts: List[str] = []
        full_texts: List[str] = []
        for svg in svg_texts:
            user_only = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT_TEXT},
                        {"type": "image"},
                    ],
                }
            ]
            with_assistant = [
                *user_only,
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": svg},
                    ],
                },
            ]
            prompt_texts.append(self.processor.apply_chat_template(user_only, add_generation_prompt=True))
            full_texts.append(self.processor.apply_chat_template(with_assistant, add_generation_prompt=False))

        # Tokenize full inputs with images (truncate to control memory)
        model_inputs = self.processor(
            images=images,
            text=full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        # Compute labels by masking out the prompt segment
        labels = model_inputs["input_ids"].clone()
        # Get prompt lengths per-sample
        prompt_tokenized = self.processor.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        # Use actual lengths before padding
        prompt_lengths: List[int] = [int(l) for l in (prompt_tokenized["attention_mask"].sum(dim=1))]
        for i, pl in enumerate(prompt_lengths):
            labels[i, :pl] = -100
        model_inputs["labels"] = labels
        return model_inputs


def build_trainer(
    data_dir: str,
    model_id: str,
    output_dir: str,
    batch_size: int,
    max_items: int,
    max_length: int,
) -> Trainer:
    dataset = SvgSftDataset(data_dir, image_size=(256, 256), max_items=max_items)

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )
    # Reduce activation memory
    model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False
    processor = AutoProcessor.from_pretrained(model_id)

    collator = Collator(processor, max_length=max_length)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        learning_rate=2e-5,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        remove_unused_columns=False,
    )

    def data_collator(features: List) -> dict:
        return collator(features)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    return trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT LLaVA on SVG reproduction pairs")
    p.add_argument("--data-dir", type=str, default="data/commons_svgs", help="Directory with .svg files")
    p.add_argument("--model-id", type=str, default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--output-dir", type=str, default="runs/sft-llava")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-items", type=int, default=64)
    p.add_argument("--max-length", type=int, default=4096, help="Max token length for text inputs")
    p.add_argument("--dry-run", action="store_true", help="Build a batch and exit without training")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Help CUDA memory behavior
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
    if args.dry_run:
        # Build dataset and processor, collate a small batch to validate shapes
        ds = SvgSftDataset(args.data_dir, image_size=(256, 256), max_items=min(2, args.max_items))
        processor = AutoProcessor.from_pretrained(args.model_id)
        collator = Collator(processor, max_length=args.max_length)
        batch = [ds[0]] if len(ds) > 0 else []
        if len(ds) > 1:
            batch.append(ds[1])
        if batch:
            out = collator(batch)
            print({k: (v.shape if hasattr(v, 'shape') else type(v)) for k, v in out.items()})
        return

    trainer = build_trainer(
        data_dir=args.data_dir,
        model_id=args.model_id,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_items=args.max_items,
        max_length=args.max_length,
    )
    trainer.train()


if __name__ == "__main__":
    main()


