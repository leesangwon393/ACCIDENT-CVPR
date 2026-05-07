import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import train_lora_qwen3vl_8b_basic as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA FT for Qwen3-VL-8B-Instruct with SGD + CosineAnnealing")
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--labels-csv", default="/workspace/yuyeon/raw/accident/sim_dataset/labels.csv")
    parser.add_argument("--video-base-path", default="/workspace/yuyeon/raw/accident/sim_dataset")
    parser.add_argument("--skip-list", default="/workspace/yuyeon/raw/accident/sim_dataset/skip_list.txt")
    parser.add_argument("--output-dir", default="/workspace/yuyeon/outputs/lora_qwen3vl8b_sgd_cosine")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nesterov", type=base.parse_bool, default=False)
    parser.add_argument("--eta-min", type=float, default=3e-6)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means run full epoch schedule")
    parser.add_argument("--max-train-samples", type=int, default=0, help="0 means use all rows")
    parser.add_argument("--train-max-frames", type=int, default=24)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=0, help="0 disables intermediate saves")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--r", type=int, default=2)
    parser.add_argument("--lora-alpha", type=int, default=4)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--finetune-vision-layers", type=base.parse_bool, default=False)
    parser.add_argument("--finetune-language-layers", type=base.parse_bool, default=True)
    parser.add_argument("--finetune-attention-modules", type=base.parse_bool, default=True)
    parser.add_argument("--finetune-mlp-modules", type=base.parse_bool, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] model={args.model_name}")
    print(f"[INFO] labels={args.labels_csv}")
    print(f"[INFO] output_dir={output_dir}")
    print(f"[INFO] optimizer=SGD momentum={args.momentum} nesterov={args.nesterov} eta_min={args.eta_min}")

    labels_df = base.load_training_rows(args)
    model, processor = base.build_model_and_processor(args)

    dataset = base.RowDataset(labels_df.to_dict("records"))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda batch: batch[0])

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = SGD(
        trainable_params,
        lr=args.learning_rate,
        momentum=args.momentum,
        nesterov=args.nesterov,
        weight_decay=args.weight_decay,
    )

    total_micro_steps = len(dataloader) * args.num_epochs
    if args.max_steps > 0:
        total_micro_steps = min(total_micro_steps, args.max_steps)
    total_optimizer_steps = max(1, math.ceil(total_micro_steps / args.gradient_accumulation_steps))
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.eta_min)

    print(f"[INFO] train_rows={len(dataset)}")
    print(f"[INFO] total_micro_steps={total_micro_steps}")
    print(f"[INFO] total_optimizer_steps={total_optimizer_steps}")

    optimizer.zero_grad(set_to_none=True)
    step = 0
    optimizer_step = 0
    progress_bar = tqdm(total=total_micro_steps, desc="FT-SGD", unit="step")

    try:
        for epoch in range(args.num_epochs):
            print(f"[INFO] starting epoch {epoch + 1}/{args.num_epochs}")
            for row in dataloader:
                if args.max_steps > 0 and step >= args.max_steps:
                    break

                messages = base.build_training_messages(pd.Series(row), args.video_base_path, args.train_max_frames)
                batch = base.build_training_features(processor, messages)
                batch = base.move_to_device(batch, model)

                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation_steps
                loss.backward()
                step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=f"{loss.detach().item() * args.gradient_accumulation_steps:.4f}")

                if step % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1

                if step % args.log_every == 0:
                    print(
                        f"[TRAIN] micro_step={step}/{total_micro_steps} "
                        f"optimizer_step={optimizer_step}/{total_optimizer_steps} "
                        f"lr={optimizer.param_groups[0]['lr']:.8f} "
                        f"loss={loss.detach().item() * args.gradient_accumulation_steps:.4f}"
                    )

                if args.save_every > 0 and optimizer_step > 0 and optimizer_step % args.save_every == 0 and step % args.gradient_accumulation_steps == 0:
                    save_dir = output_dir / f"step_{optimizer_step}"
                    model.save_pretrained(save_dir)
                    processor.save_pretrained(save_dir)
                    print(f"[INFO] checkpoint saved -> {save_dir}")

            if args.max_steps > 0 and step >= args.max_steps:
                break
    finally:
        progress_bar.close()

    final_dir = output_dir / "final"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"[INFO] final adapter saved -> {final_dir}")


if __name__ == "__main__":
    main()
