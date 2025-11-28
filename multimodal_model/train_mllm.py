# multimodal_model/train_mllm.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import os
import pandas as pd
import torchvision.transforms as transforms
import random

from utils.training_utils import get_device
from vision_transformer.vit import ViT
from language_model.llm import GPTModel
from language_model.tokenizer import CharacterTokenizer
from .connector import Connector
from .mllm import MLLM
from datasets.Flickr8k import Flickr8kDataset

# ... (helper functions remain the same) ...


def generate_caption_for_sample(mllm_model, tokenizer, image, device, max_len=50):
    mllm_model.eval()
    image = image.unsqueeze(0).to(device)
    start_prompt = ''
    with torch.no_grad():
        generated_caption = mllm_model.generate(
            image=image, prompt=start_prompt, max_new_tokens=max_len, temperature=0.8, top_k=10)
    return generated_caption


def create_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])


def build_tokenizer(corpus_path, save_path):
    if os.path.exists(save_path):
        print(f"Loading existing tokenizer from {save_path}")
        tokenizer = CharacterTokenizer(corpus=None)
        tokenizer.load_vocab(save_path)
    else:
        print(f"Building tokenizer from {corpus_path}...")
        df = pd.read_csv(corpus_path)
        corpus = "\n".join(df['caption'].tolist())
        tokenizer = CharacterTokenizer(corpus)
        tokenizer.save_vocab(save_path)
        print(f"Tokenizer saved to {save_path}")
    return tokenizer


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    loop = tqdm(dataloader, leave=True, desc=f"Epoch {epoch+1} [Train]")

    for images, captions in loop:
        images, captions = images.to(device), captions.to(device)
        optimizer.zero_grad(set_to_none=True)

        model_input_text = captions[:, :-1]
        targets = captions[:, 1:]

        logits, num_visual_tokens = model(images, model_input_text)

        labels = torch.full((logits.shape[0], logits.shape[1]),
                            criterion.ignore_index, device=device, dtype=torch.long)
        label_start_idx = num_visual_tokens
        label_end_idx = num_visual_tokens + targets.shape[1]
        labels[:, label_start_idx:label_end_idx] = targets

        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)


def train(config):
    """
    The main training function for the Multimodal Large Language Model (MLLM).
    """
    # --- 0. Setup ---
    device = get_device(config['training']['device'])
    train_cfg = config['training']
    paths_cfg = config['paths']
    model_cfg = config['model']
    print("="*50)
    print("INFO: Using robust Flickr8kDataset implementation.")
    print("="*50)

    # --- 1. Build or Load Tokenizer ---
    tokenizer = build_tokenizer(
        paths_cfg['captions_corpus_path'], paths_cfg['tokenizer_save_path'])
    vocab_size = tokenizer.get_vocab_size()
    pad_token_id = tokenizer.get_pad_token_id()

    # --- 2. Initialize Models ---
    vision_encoder = ViT(
        img_size=model_cfg['vision_encoder']['image_size'],
        patch_size=model_cfg['vision_encoder']['patch_size'],
        in_channels=model_cfg['vision_encoder']['in_channels'],
        d_model=model_cfg['vision_encoder']['vision_dim'],
        num_layers=model_cfg['vision_encoder']['n_layers'],
        n_heads=model_cfg['vision_encoder']['n_heads'],
        d_ff=model_cfg['vision_encoder']['d_ff'],
        dropout=model_cfg['dropout']
    ).to(device)

    language_model = GPTModel(
        vocab_size=vocab_size,
        d_model=model_cfg['language_model']['language_dim'],
        num_layers=model_cfg['language_model']['n_layers'],
        n_heads=model_cfg['language_model']['n_heads'],
        d_ff=model_cfg['language_model']['d_ff'],
        max_len=model_cfg['language_model']['max_len'],
        dropout=model_cfg['dropout']
    ).to(device)

    connector = Connector(
        vision_dim=model_cfg['vision_encoder']['vision_dim'],
        language_dim=model_cfg['language_model']['language_dim'],
        connector_type=model_cfg['connector']['type']
    ).to(device)

    mllm = MLLM(vision_encoder, language_model,
                connector, tokenizer).to(device)
    mllm.freeze_parameters(train_cfg['freeze_vit'], train_cfg['freeze_llm'])

    # --- 3. Prepare Data ---
    transform = create_transform(model_cfg['vision_encoder']['image_size'])
    full_dataset = Flickr8kDataset(
        root=config['data']['data_root'], captions_file="captions.txt", transform=transform, tokenizer=tokenizer)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size])
    print(
        f"Dataset split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    def collate_fn(batch):
        images, captions = zip(*batch)
        images = torch.stack(images, 0)
        captions_padded = pad_sequence(
            captions, batch_first=True, padding_value=pad_token_id)
        return images, captions_padded

    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg['batch_size'], shuffle=True, num_workers=train_cfg['num_workers'], collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg['batch_size'],
                            shuffle=False, num_workers=train_cfg['num_workers'], collate_fn=collate_fn)
    sample_img, sample_gt_tensor = random.choice(val_dataset)
    print("A fixed sample image has been chosen for generation during training.")

    # --- 4. Setup Optimizer and Loss ---
    optimizer = optim.Adam(mllm.parameters(), lr=train_cfg['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    # --- 5. Training Loop ---
    best_val_loss = float('inf')
    print("Starting MLLM training...")

    for epoch in range(train_cfg['epochs']):
        avg_train_loss = train_one_epoch(
            mllm, train_loader, optimizer, criterion, device, epoch)

        # --- Evaluation Phase ---
        if (epoch + 1) % train_cfg['eval_interval'] == 0:
            mllm.eval()
            total_val_loss = 0
            with torch.no_grad():
                val_loop = tqdm(
                    val_loader, leave=True, desc=f"Epoch {epoch+1}/{train_cfg['epochs']} [Eval]")
                for images, captions in val_loop:
                    images, captions = images.to(device), captions.to(device)
                    model_input_text = captions[:, :-1]
                    targets = captions[:, 1:]

                    logits, num_visual_tokens = mllm(images, model_input_text)

                    labels = torch.full(
                        (logits.shape[0], logits.shape[1]), criterion.ignore_index, device=device, dtype=torch.long)
                    label_start_idx = num_visual_tokens
                    label_end_idx = num_visual_tokens + targets.shape[1]
                    labels[:, label_start_idx:label_end_idx] = targets

                    loss = criterion(
                        logits.view(-1, logits.size(-1)), labels.view(-1))

                    total_val_loss += loss.item()
                    val_loop.set_postfix(loss=loss.item())

            avg_val_loss = total_val_loss / len(val_loader)
            print(
                f"Epoch {epoch+1}/{train_cfg['epochs']} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            # ... (Sample generation and checkpointing remain the same) ...
            print("\n--- Generating Sample Caption ---")
            gt_caption_full = tokenizer.decode(sample_gt_tensor.tolist())
            gt_caption_clean = (gt_caption_full.replace(tokenizer.sos_token, '').replace(
                tokenizer.eos_token, '').replace(tokenizer.pad_token, '').strip())
            generated_caption = generate_caption_for_sample(
                mllm_model=mllm, tokenizer=tokenizer, image=sample_img, device=device, max_len=100)
            print(f"  Ground Truth: {gt_caption_clean}")
            print(f"  Generated:    {generated_caption}")
            print("--- End of Sample ---\n")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                os.makedirs(os.path.dirname(
                    paths_cfg['best_model_save_path']), exist_ok=True)
                torch.save(mllm.state_dict(),
                           paths_cfg['best_model_save_path'])
                print(
                    f"New best model saved to {paths_cfg['best_model_save_path']} with Val Loss: {avg_val_loss:.4f}\n")

    print("Training finished.")


if __name__ == "__main__":
    import argparse
    from utils.config_parser import parse_config

    parser = argparse.ArgumentParser(description="Train MLLM (ViT + GPT)")
    parser.add_argument("--config", type=str, default="configs/mllm_config.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()

    cfg = parse_config(args.config)
    train(cfg)
