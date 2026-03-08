from __future__ import annotations

import argparse

import unsloth  # noqa: F401
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def format_row(example: dict) -> dict:
    return {"text": example["prompt"] + "\n\nJSON:\n" + example["completion"]}


def load_model_with_lora(model_name: str, max_seq_length: int, load_in_4bit: bool):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=7,
    )
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--dev_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", type=str, default="outputs/sft_overseer")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--load_in_4bit", action="store_true")
    args = parser.parse_args()

    train_ds = load_dataset("json", data_files=args.train_file)["train"].map(format_row)
    dev_ds = load_dataset("json", data_files=args.dev_file)["train"].map(format_row)

    model, tokenizer = load_model_with_lora(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        args=SFTConfig(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=20,
            save_steps=20,
            save_total_limit=2,
            bf16=True,
            report_to="none",
            dataset_text_field="text",
            max_length=args.max_seq_length,
        ),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
