import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
import pdb
import torchvision.transforms as transforms
import collections
import unicodedata
import re
import random
import math

import time
import sys
import logging
from datetime import datetime
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

#CONFIG
save_path = "dtrocr_model.pth"
#GPT2_MODEL = "gpt2"
GPT2_MODEL = 'surajp/gpt2-hindi'
#IMAGE_SIZE = (256, 96)
IMAGE_SIZE = (256, 64) #width, height
#MODEL_MODE = "TRAIN" #"TRAIN" or "INFERENCE"
MODEL_MODE = "INFERENCE"
#MODEL_MODE = "TRAIN"
CONFIG_NUM_EPOCHS = 50
CONFIG_BATCH_SIZE = 32 #out of memory for 128
CONFIG_NUM_IMAGES = 65000 #number of images to train on
CHECKPOINT_DIR = "checkpoints"
STEP_SAVE_INTERVAL = 50  # save every N training steps

training_idx_start = 0
#training_idx_end = training_idx_start + 20000
training_idx_end = training_idx_start + CONFIG_NUM_IMAGES

# Visual encoder toggle (decoder-only if False)
USE_VISUAL_ENCODER = False
# Number of transformer layers in visual encoder (only used if enabled)
VISUAL_ENCODER_LAYERS = 2
VISUAL_ENCODER_NHEAD = 12
VISUAL_ENCODER_FF_DIM = 3072
VISUAL_DROPOUT = 0.1

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---------- Unified Logging (console + file, captures print and logging) ----------
# Small helper to format elapsed seconds as d h m s for readability
def format_elapsed_time(total_seconds: float) -> str:
    secs = int(total_seconds)
    days, remainder = divmod(secs, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    if minutes or hours or days:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)

# Create timestamped log file in the same directory as this script
_script_stem = Path(__file__).stem
_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
_script_dir = os.path.dirname(os.path.abspath(__file__))
_logs_dir = os.path.join(_script_dir, "LOGS")
os.makedirs(_logs_dir, exist_ok=True)
_log_file_path = os.path.join(_logs_dir, f"{_script_stem}_{_timestamp}.log")

# Configure root logger to log to both file and console
_root_logger = logging.getLogger()
# Clear existing handlers to avoid duplicate logs if re-imported
for _h in list(_root_logger.handlers):
    _root_logger.removeHandler(_h)
_root_logger.setLevel(logging.DEBUG)
_log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

_file_handler = logging.FileHandler(_log_file_path, encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(_log_formatter)

# Ensure console output still shows by writing to the original stdout stream
_console_handler = logging.StreamHandler(stream=sys.__stdout__)
_console_handler.setLevel(logging.DEBUG)
_console_handler.setFormatter(_log_formatter)

_root_logger.addHandler(_file_handler)
_root_logger.addHandler(_console_handler)


class _StreamToLogger:
    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self._buffer = ""

    def write(self, message: str):
        # Buffer until newline to avoid splitting lines mid-print
        if not isinstance(message, str):
            message = str(message)
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line:
                self.logger.log(self.level, line)

    def flush(self):
        if self._buffer:
            self.logger.log(self.level, self._buffer)
            self._buffer = ""


# Redirect stdout/stderr so that print() and errors go through logging
sys.stdout = _StreamToLogger(_root_logger, logging.INFO)
sys.stderr = _StreamToLogger(_root_logger, logging.ERROR)


# ---------- Checkpointing Helpers ----------
LAST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "last.pt")


def save_checkpoint(model, optimizer, epoch, global_step, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path} (epoch={epoch}, step={global_step})")


def load_checkpoint(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "rng_state" in checkpoint:
        rng_state = checkpoint["rng_state"]
        if isinstance(rng_state, torch.Tensor) and rng_state.dtype == torch.uint8:
            rng_state = rng_state.type(torch.ByteTensor)
        torch.set_rng_state(rng_state)
    if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
        rng_state_all = checkpoint["cuda_rng_state_all"]
        if(len(rng_state_all)==1):
            rng_state_all = rng_state_all[0]
        rng_state_all=rng_state_all.type(torch.ByteTensor)
        torch.cuda.set_rng_state_all([rng_state_all])
    epoch = checkpoint.get("epoch", 0)
    global_step = checkpoint.get("global_step", 0)
    print(f"Loaded checkpoint from {path}: epoch={epoch}, step={global_step})")
    return epoch, global_step


def load_alphabet(alphabet_file):
    """
    Load alphabet/vocabulary from file
    
    Args:
        alphabet_file: path to alphabet file (one character per line)
        
    Returns:
        list of characters
    """
    alphabet = []
    with open(alphabet_file, 'r', encoding='utf-8') as f:
        for line in f:
            char = line.strip()
            if char:  # Skip empty lines
                alphabet.append(char)
    
    print(f"Loaded {len(alphabet)} characters from {alphabet_file}")
    return alphabet


class TextDataset(Dataset):
    def __init__(self, labels_file_name, relative_path_to_images, vocabulary=None, 
                 clean_labels=True, encoding='utf-8', idx_start=None, idx_end=None):
        """
        Args:
            labels_file_name: path to file with image_name label pairs
            relative_path_to_images: path prefix for images
            vocabulary: list/string of valid characters (optional)
            clean_labels: whether to remove out-of-vocabulary characters
            encoding: file encoding
        """
        if idx_start is None:
            raise ValueError("idx_start must be provided")
        if idx_end is None:
            raise ValueError("idx_end must be provided")

        self.images = []
        self.labels = []
        self.vocabulary = vocabulary
        self.clean_labels = clean_labels
        self.idx_start = idx_start
        self.idx_end = idx_end
        
        # Read data file
        with open(labels_file_name, 'r', encoding=encoding) as f:
            for line_no, line in enumerate(f, 1):
                parts = line.strip().split(' ', 1)
                if len(parts) < 2:
                    print(f"Warning: Line {line_no} has invalid format: {line.strip()}")
                    continue
                    
                image_name, label_text = parts
                self.images.append(relative_path_to_images + image_name)
                
                # Clean labels if vocabulary is provided
                if self.clean_labels and self.vocabulary:
                    cleaned_label = self._clean_label(label_text)
                    if cleaned_label != label_text and cleaned_label:
                        print(f"Cleaned label: '{label_text}' -> '{cleaned_label}'")
                    self.labels.append(cleaned_label if cleaned_label else label_text)
                else:
                    self.labels.append(label_text)
        
        print(f"Loaded {len(self.images)} samples from {labels_file_name}")
    
    def _clean_label(self, label):
        """Remove characters not in vocabulary"""
        if not self.vocabulary:
            return label
        
        # Create regex pattern for out-of-vocabulary characters
        voc_str = ''.join(re.escape(c) for c in self.vocabulary)
        out_of_vocab = f'[^{voc_str}]'
        
        # Check if cleaning is needed
        to_remove_chars = re.findall(out_of_vocab, label) #just to print
        if to_remove_chars:
            print(f"Characters to remove from label '{label}': {to_remove_chars}")
            with open("removed_label_characters.log", "a", encoding="utf-8") as log_f:
                log_f.write(f"Characters to remove from label '{label}': {to_remove_chars}\n")

        to_remove = re.search(out_of_vocab, label)
        if to_remove:
            pattern = re.compile(out_of_vocab)
            cleaned = pattern.sub('', label)
            return cleaned
        
        return label
    
    def __getitem__(self, idx):
        # Load & preprocess image
        try:
            imgPath = self.images[self.idx_start + idx]
            #print(f"Loading image: {imgPath}") #log verbose logging
            img = cv2.imread(imgPath)
            if img is None:
                print(f"Failed to load image: {imgPath}") #log verbose logging
                return None
            
            # Resize to (width=256, height=96)
            img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
            
            # Convert to (channels, height, width) format
            img = img.transpose(2, 0, 1)  # HWC -> CHW
            
            # Normalize to [-1, 1]
            img = torch.from_numpy(img).float() / 255.0
            img = (img - 0.5) / 0.5
            
        except Exception as e:
            print(f'Exception loading image {self.images[idx]}: {e}')
            return None
        
        # Return image and raw text label
        # Encoding will be done by the converter in the training loop
        label_text = self.labels[idx]
        
        return img, label_text
    
    def __len__(self):
        #return len(self.images)
        return self.idx_end - self.idx_start  # Remove hardcoded limit


# Create dataset
alphabet = load_alphabet('alphabet/hi.txt')

training_dataset = TextDataset(
        labels_file_name='IIIT-HW-Hindi_v1/train.txt',
        relative_path_to_images='IIIT-HW-Hindi_v1/HindiSeg//',
        vocabulary=alphabet,  # Pass vocabulary for label cleaning
        clean_labels=True,    # Clean out-of-vocabulary characters
        encoding='utf-8',
        idx_start = training_idx_start,
        idx_end = training_idx_end
    )

# Collate function for DataLoader
def collate_fn(batch):
    """
    Collate function to handle batching of images and text labels
    Filters out None values from failed image loads
    """
    # Filter out None values
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None, None
    
    try:
        images, label_texts = zip(*batch)
        # Stack images into a batch
        images = torch.stack(images, 0)
        return images, label_texts
    except Exception as e:
        print(f'Collate exception: {e}')
        return None, None


loader = DataLoader(training_dataset, batch_size=CONFIG_BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)

# --- 2D sinusoidal positional embeddings for visual tokens ---
def _get_1d_sincos_pos_embed(embed_dim: int, length: int, device):
    position = torch.arange(length, device=device).float()
    div_term = torch.exp(torch.arange(0, embed_dim, 2, device=device).float() * (-math.log(10000.0) / embed_dim))
    pe = torch.zeros(length, embed_dim, device=device)
    pe[:, 0::2] = torch.sin(position[:, None] * div_term)
    pe[:, 1::2] = torch.cos(position[:, None] * div_term)
    return pe  # [L, D]

def build_2d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int, device):
    assert embed_dim % 2 == 0
    half_dim = embed_dim // 2
    pe_h = _get_1d_sincos_pos_embed(half_dim, grid_h, device)  # [H, D/2]
    pe_w = _get_1d_sincos_pos_embed(half_dim, grid_w, device)  # [W, D/2]
    pe_h = pe_h[:, None, :].expand(grid_h, grid_w, half_dim)   # [H, W, D/2]
    pe_w = pe_w[None, :, :].expand(grid_h, grid_w, half_dim)   # [H, W, D/2]
    pe = torch.cat([pe_h, pe_w], dim=-1).reshape(1, grid_h * grid_w, embed_dim)  # [1, H*W, D]
    return pe

# --- Copy the Model Class Here (from the explanation above) ---
class DTrOCR(nn.Module):
    def __init__(self, gpt2_model_name=GPT2_MODEL, patch_size=(4, 8)): 
        super().__init__()
        self.decoder = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        self.use_visual_encoder = USE_VISUAL_ENCODER
        
        # Force the model to resize its embeddings to match the tokenizer's vocabulary
        # This prevents the "Index out of range" error.
        self.decoder.resize_token_embeddings(len(tokenizer))# for IndexError: index out of range in self error produced by embedding layer

        # Note: Patch size (H, W). Paper uses 8x4, but code usually expects HxW. 
        # Ensure this matches your image resizing logic.
        self.patch_embedding = nn.Conv2d(3, 768, kernel_size=patch_size, stride=patch_size)
        if self.use_visual_encoder and VISUAL_ENCODER_LAYERS > 0:
            self.visual_ln = nn.LayerNorm(768)
            self.visual_dropout = nn.Dropout(VISUAL_DROPOUT)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=768, nhead=VISUAL_ENCODER_NHEAD, dim_feedforward=VISUAL_ENCODER_FF_DIM, dropout=VISUAL_DROPOUT, batch_first=True
            )
            self.visual_encoder = nn.TransformerEncoder(encoder_layer, num_layers=VISUAL_ENCODER_LAYERS)
        else:
            self.visual_ln = None
            self.visual_dropout = None
            self.visual_encoder = None
        self.sep_token_emb = nn.Parameter(torch.randn(1, 1, 768))

    def forward_features(self, images):
        """Helper to get just the visual embeddings + SEP"""
        patches = self.patch_embedding(images)  # [B, 768, H_p, W_p]
        b, c, h_p, w_p = patches.shape
        visual_embeds = patches.flatten(2).transpose(1, 2)  # [B, H_p*W_p, 768]
        pos_embed = build_2d_sincos_pos_embed(768, h_p, w_p, device=visual_embeds.device)  # [1, H*W, 768]
        visual_embeds = visual_embeds + pos_embed
        if self.visual_encoder is not None:
            visual_embeds = self.visual_ln(visual_embeds)
            visual_embeds = self.visual_dropout(visual_embeds)
            visual_embeds = self.visual_encoder(visual_embeds)
        sep_token = self.sep_token_emb.expand(images.shape[0], -1, -1)
        return torch.cat([visual_embeds, sep_token], dim=1)

    def forward(self, images, text_input_ids):
        # Training forward pass
        # text_embeds = self.decoder.transformer.wte(text_input_ids)
        
        embedding_layer = self.decoder.get_input_embeddings()
        text_embeds = embedding_layer(text_input_ids)

        visual_part = self.forward_features(images)
        combined_embeds = torch.cat([visual_part, text_embeds], dim=1)
        return self.decoder(inputs_embeds=combined_embeds)

# --- Setup ---
# Initialize Tokenizer and Model
tokenizer = GPT2Tokenizer.from_pretrained(GPT2_MODEL)

# Ensure PAD is distinct from EOS so EOS can be learned as an end signal
if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

# Ensure a pad token exists for batching with padding
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


deviceName = "cuda" if torch.cuda.is_available() else "cpu"
#deviceName = "cpu"

print(f"Using device: {deviceName}")
device = torch.device(deviceName)
#inference_device = torch.device("cpu")
inference_device = device

# Deterministic settings for reproducible inference
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model = DTrOCR().to(device) # Assuming DTrOCR class is defined
# Align model pad token id with tokenizer
model.decoder.config.pad_token_id = tokenizer.pad_token_id
optimizer = optim.AdamW(model.parameters(), lr=1e-4)



def Train():
    start_epoch = 0
    global_step = 0

    # --- 4. The Training Loop ---
    print("Starting training...")
    model.train()
    
    if os.path.exists(LAST_CKPT_PATH):
        try:
            print(f"Loading checkpoint from {LAST_CKPT_PATH}")
            start_epoch, global_step = load_checkpoint(model, optimizer, LAST_CKPT_PATH, device)
            print(f"Loaded checkpoint from {LAST_CKPT_PATH}: epoch={start_epoch}, step={global_step}")
        except Exception as e:
            print(f"Failed to load checkpoint '{LAST_CKPT_PATH}': {e}")
    
    # Local state for logging timings (avoid setting attributes on builtins)
    log_last_time = None
    log_start_time = None
    try:
        print(f"Training for {CONFIG_NUM_EPOCHS} epochs")
        for epoch in range(start_epoch, CONFIG_NUM_EPOCHS): # Run for 5 epochs
            for batch_idx, (images, label_strings) in enumerate(loader):
                if images is None: 
                    continue # Skip empty batches
                
                # Move images to GPU
                images = images.to(device)
                
                # --- THE BRIDGE: Convert Raw Text to Token IDs ---
                # 1) Tokenize WITHOUT special tokens
                tok = tokenizer(
                    list(label_strings),
                    padding=True,
                    return_tensors="pt",
                    add_special_tokens=False
                )
                tok = {k: v.to(device) for k, v in tok.items()}

                # 2) Append EOS to each sequence explicitly, create new padded tensor
                pad_token_id = tokenizer.pad_token_id
                eos_token_id = tokenizer.eos_token_id
                B, T = tok["input_ids"].shape
                new_T = T + 1
                text_input_ids = torch.full((B, new_T), pad_token_id, dtype=torch.long, device=device)
                text_input_ids[:, :T] = tok["input_ids"]
                lengths = tok["attention_mask"].sum(dim=1)  # [B]
                for b in range(B):
                    end_pos = int(lengths[b].item())
                    if end_pos < new_T:
                        text_input_ids[b, end_pos] = eos_token_id

                # --- Reset Gradients ---
                optimizer.zero_grad()

                # --- Build shifted decoder inputs (teacher forcing) ---
                # Use BOS if available, otherwise EOS as start token
                start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
                pad_token_id = tokenizer.pad_token_id
                text_len = text_input_ids.shape[1]

                # Shift-right: decoder_input_ids = [START] + text[:-1]
                decoder_input_ids = text_input_ids.clone()
                decoder_input_ids[:, 1:] = text_input_ids[:, :-1]
                decoder_input_ids[:, 0] = start_token_id

                # Get embeddings for shifted decoder inputs
                embedding_layer = model.decoder.get_input_embeddings()
                text_embeds_shifted = embedding_layer(decoder_input_ids)

                # --- Forward Pass with visual prefix + shifted text ---
                visual_part = model.forward_features(images)
                combined_embeds = torch.cat([visual_part, text_embeds_shifted], dim=1)
                outputs = model.decoder(inputs_embeds=combined_embeds)
                logits = outputs.logits  # [B, prefix+T, V]

                # --- Calculate Loss only on text positions (next-token prediction) ---
                logits_text = logits[:, -text_len:, :]  # keep last T positions
                labels = text_input_ids.clone()
                labels[text_input_ids == pad_token_id] = -100

                loss = torch.nn.functional.cross_entropy(
                    logits_text.reshape(-1, logits_text.size(-1)),
                    labels.reshape(-1)
                )
                
                # --- Backward Pass ---
                loss.backward()
                optimizer.step()

                global_step += 1
                
                # --- Logging ---
                if batch_idx % 10 == 0:
                    if log_last_time is None:
                        log_last_time = time.time()
                        log_start_time = log_last_time
                        elapsed_since_last = 0.0
                        total_elapsed = 0.0
                    else:
                        now = time.time()
                        elapsed_since_last = now - log_last_time
                        total_elapsed = now - log_start_time
                        log_last_time = now
                    print(f"Epoch [{epoch+1}] Batch [{batch_idx}/{len(loader)}] Loss: {loss.item():.4f} | Time since last: {elapsed_since_last:.2f}s | Total elapsed: {total_elapsed:.2f}s")
                    #if loss.item() <= 0.0001:
                    #    print(f"Stopping training as loss <= 0.0001 at step {global_step}.")
                    #    raise StopIteration("Stopping criterion met: loss <= 0.0001")

                # --- Periodic checkpoint ---
                if STEP_SAVE_INTERVAL and (global_step % STEP_SAVE_INTERVAL == 0):
                    print(f"Saving checkpoint at step {global_step} {time.time()}")
                    save_checkpoint(model, optimizer, epoch, global_step, LAST_CKPT_PATH)
                    print(f"Checkpoint saved at step {global_step} {time.time()}")

            # Save checkpoint at end of each epoch
            print(f"EPOCH END. Saving checkpoint at step {global_step} {time.time()}")
            save_checkpoint(model, optimizer, epoch + 1, global_step, LAST_CKPT_PATH)
            print(f"EPOCH END. Checkpoint saved at step {global_step} {time.time()}")

            # Ensure to bring model to CPU before saving to avoid CUDA device-side asserts
            print(f"EPOCH END. Saving model at epoch {epoch} {time.time()}")
            model_cpu = model.to('cpu')
            torch.save(model_cpu.state_dict(), save_path + "_epoch_" + str(epoch)) #save model after every epoch
            model.to(device) 
            print(f"EPOCH END. saved model at epoch {epoch} {time.time()}")

    except KeyboardInterrupt:
        print("Interrupted. Saving checkpoint before exit...")
        try:
            #save_checkpoint(model, optimizer, epoch, global_step, LAST_CKPT_PATH)
            pass
        except Exception as e:
            print(f"Failed to save checkpoint on interrupt: {e}")
        raise
    except StopIteration:
        print("Stopping training as loss <= 0.0001.")
    except Exception as e:
        print(f"Error in training loop: {e}")
        
    # Ensure to bring model to CPU before saving to avoid CUDA device-side asserts
    model_cpu = model.to('cpu')
    torch.save(model_cpu.state_dict(), save_path) #directly saving the model was giving cuda sssert
    model.to(device)  # Move back to original device if needed for further use

    print(f"Model saved to {save_path}")
    total_time = None
    if log_start_time is not None:
        total_time = time.time() - log_start_time
        human_time = format_elapsed_time(total_time)
        print(f"Training done. Model saved to {save_path}. Total elapsed time: {human_time} ({total_time:.2f} seconds)")
    else:
        print(f"Training done. Model saved to {save_path}")
    
    #test with a sample image
    InferenceInternal(model, tokenizer)

def preprocess_image(image_path):
    """
    Loads and resizes image to match the training format.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Failed to load image. Check format.")

    # Resize to fixed size (Width, Height) -> (128, 32)
    # Note: cv2.resize takes (Width, Height)
    img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    # Convert to CHW format (Channels, Height, Width)
    img = img.transpose(2, 0, 1)

    # Normalize to range [-1, 1]
    #img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).float() / 255.0
    img = (img - 0.5) / 0.5

    # Convert to Tensor and add Batch dimension [1, 3, 32, 128]
    if isinstance(img, torch.Tensor):
        tensor = img.unsqueeze(0)
    else:
        tensor = torch.from_numpy(img).unsqueeze(0)
    return tensor.to(device)


def predict_text(model, tokenizer, image_path, max_length=20):
    """
    Generates text from an image using Greedy Decoding.
    """
    model.eval() # Set to evaluation mode
    
    # Preprocess image
    try:
        image_tensor = preprocess_image(image_path)
    except Exception as e:
        return f"Error: {e}"

    with torch.no_grad():
        # Step A: Get the visual context (The "Prompt")
        # This creates the embeddings for [Image Patches] + [SEP]
        current_embeddings = model.forward_features(image_tensor)

        # Prepend a start token to kick off decoding
        start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        start_token = torch.tensor([[start_token_id]], device=image_tensor.device, dtype=torch.long)
        start_embed = model.decoder.transformer.wte(start_token)
        current_embeddings = torch.cat([current_embeddings, start_embed], dim=1)

        generated_ids = []

        # Step B: Autoregressive Generation Loop (deterministic greedy decoding)
        for _ in range(max_length):
            outputs = model.decoder(inputs_embeds=current_embeddings)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            if next_token_id.item() == tokenizer.eos_token_id:
                break

            generated_ids.append(next_token_id.item())
            next_token_embed = model.decoder.transformer.wte(next_token_id)
            current_embeddings = torch.cat([current_embeddings, next_token_embed], dim=1)

    # Step C: Decode IDs to Text
    decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return decoded_text


def Inference():
    print(f"Loading model on {inference_device}...")
    
    # Initialize Tokenizer & Model
    tokenizer = GPT2Tokenizer.from_pretrained(GPT2_MODEL)
    model = DTrOCR().to(inference_device)
    
    # Load Weights
    if os.path.exists(save_path):
        print(f"Loading weights from {save_path}")
        device_map = torch.device(inference_device)
        model.load_state_dict(torch.load(save_path, map_location=device_map))
        model.to(inference_device)
    else:
        print(f"Error: {save_path} not found.")
        return
    
    #InferenceInternal(model, tokenizer)
    #InferenceTestData(model, tokenizer)
    InferenceTrainData(model, tokenizer)
    InferenceTestData(model, tokenizer)
    InferenceValtData(model, tokenizer)

def InferenceInternal(model, tokenizer):
    # Example Usage
    # Replace this with the path to your image
    #test_image_path = "C:/Users/j/Desktop/cs230/Patch_GPT/files_to_be_copied/IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1/HindiSeg/HindiSeg/train/7/1/2.jpg"
    #test_image_path = "C:/Users/j/Desktop/cs230/Patch_GPT/files_to_be_copied/IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1/HindiSeg/HindiSeg/train/7/1/1.jpg"
    test_image_path = "IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1/HindiSeg/train/8/249/20.jpg"
    images=['IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/249/20.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/231/37.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/1/117/4.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/7/142/31.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/7/100/14.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/1/170/11.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/197/19.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/153/19.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/53/36.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/289/33.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/18/38.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/86/28.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/7/135/39.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/7/172/19.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/207/27.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/95/1.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/68/30.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/150/28.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/40/17.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/222/6.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/66/3.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/257/36.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/35/29.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/239/30.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/17/32.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/7/118/31.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/202/19.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/138/19.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/141/24.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/162/32.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/72/30.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/257/7.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/184/12.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/117/33.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/151/9.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/76/3.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/94/31.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/21/14.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/191/38.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/1/222/32.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/7/199/28.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/183/26.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/233/30.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/168/15.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/107/7.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/133/29.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/1/200/15.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/102/15.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/7/50/26.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/200/13.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/77/12.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/1/76/9.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/1/76/7.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/1/6/36.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/253/22.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/251/22.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/249/7.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/290/11.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/107/38.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/218/18.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/159/6.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/207/22.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/256/33.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/195/26.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/106/28.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/1/85/4.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/237/30.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/13/22.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/1/168/36.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/37/9.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/44/29.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/29/24.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/15/3.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/7/17/22.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/69/26.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/234/36.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/77/9.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/111/22.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/7/142/14.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/168/29.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/7/8/2.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/47/30.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/128/23.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/145/9.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/1/62/13.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/146/33.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/107/30.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/85/11.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/246/26.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/16/18.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/6/5.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/215/5.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/186/12.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/269/9.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/174/11.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/7/31/29.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/298/32.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/47/7.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/161/4.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/7/106/16.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/32/25.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/261/17.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/67/27.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/7/30/5.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/170/33.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/184/29.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/263/4.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/201/34.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/5/14.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/7/160/3.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/7/51/18.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/92/4.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/7/82/16.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/14/6.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/309/16.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/63/8.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/209/40.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/98/15.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/254/1_7.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/202/25.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/7/212/9.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/54/8.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/12/7.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/7/159/8.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/1/175/11.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/69/40.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/254/17.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/59/24.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/1/224/3.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/79/28.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/105/9.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/44/3.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/105/3.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/60/19.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/190/14.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/243/20.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/140/15.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/18/27_1.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/310/13.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/1/121/16.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/155/22.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/108/11.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/7/197/27.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/126/33.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/42/30.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/95/34.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/2/24.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/239/23.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/7/4/15.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/285/30.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/1/40/4.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/1/25/35.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/148/36.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/1/86/12.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/82/18.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/239/27.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/230/8.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/210/22.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/58/18.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/60/33.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/161/17.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/2/5.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/213/26.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/68/20.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/286/16.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/166/13.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/80/5.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/7/138/9.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/118/18.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/190/37.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/102/29.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/289/16.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/203/28.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/7/142/36.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/10/25.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/109/35.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/117/20.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/118/7.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/41/11.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/301/12.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/7/66/28.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/143/5.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/23/1.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/114/2.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/7/21/26.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/244/20.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/19/3.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/241/26.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/195/8.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/7/55/21.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/92/25.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/7/195/21.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/7/157/27.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/35/3.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/1/233/29.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/7/94/28.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/46/31.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/83/15.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/133/27.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/166/12.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/3/30.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/1/31/26.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/249/9.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/129/12.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/7/17/26.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/36/23.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/25/34.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/136/10.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/17/20.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/94/30.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/28/1.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/65/31.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/182/32.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/143/37.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/14/24.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/42/12.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/7/154/28.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/6/29.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/270/17.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/1/220/11.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/220/3.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/47/22.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/145/31.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/74/24.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/1/205/7.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/58/15.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/27/6.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/1/21/23.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/1/167/28.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/140/9.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/139/14.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/276/12.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/1/129/18.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/244/30.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/198/33.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/177/21.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/51/23.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/281/32.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/118/21.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/159/30.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/8/5.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/205/29.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/188/21.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/1/24/13.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/65/16.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/284/24.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/2/22.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/91/10.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/2/16.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/224/29.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/77/37.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/183/34.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/228/3.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/48/17.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/1/158/18.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/150/12.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/23/13.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/26/36.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/153/12.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/172/4.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/99/3.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/179/23.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/140/2.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/83/2.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/147/14.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/58/27.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/18/17.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/216/23.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/33/35.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/33/9.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/303/13.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/7/51/16.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/22/22.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/157/28.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/7/35/26.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/14/26.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/46/36.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/7/13/20.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/91/14.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/80/12.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/94/40.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/82/29.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/38/11.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/213/21.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/103/11.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/1/160/21.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/114/21.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/63/34.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/166/35.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/1/12/12.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/1/5/31.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/7/107/35.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/118/19.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/60/18.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/7/27/30.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/26/33.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/99/6.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/1/98/5.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/272/6.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/4/124/9.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/171/31.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/93/12.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/217/20.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/145/37.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/1/104/12.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/239/5.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/7/106/9.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/66/34.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/229/2.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/23/24.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/1/158/13.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/2/70/4.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/8/55/38.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/7/97/21.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/1/205/18.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/237/21.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/288/12.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/10/149/15.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/168/2.jpg','IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1//HindiSeg/train/5/252/40.jpg']

    if not os.path.exists(test_image_path):
        print("image does not exist")
        return
    
    print(f"Reading image: {test_image_path}")
    result = predict_text(model, tokenizer, test_image_path)
    print(f"\nRecognized Text: '{result}'")
     
     #add pandas data frame to store the results
    results_df = pd.DataFrame(columns=['ground_truth', 'predicted_text', 'image_path'])

    for image_path in images:
        if not os.path.exists(test_image_path):
            print("image does not exist")
            return
        result = predict_text(model, tokenizer, image_path)

        # Remove the prefix from image_path to get the lookup key
        prefix = 'IIIT-HW-Hindi_v1.tar/IIIT-HW-Hindi_v1/'
        if image_path.startswith(prefix):
            image_key = image_path[len(prefix):]
        else:
            image_key = image_path
        
        # Remove any "/" at the beginning of image_key
        image_key = image_key.lstrip('/')

        ground_truth = ""
        train_txt_path = r"C:\Users\j\Desktop\cs230\Patch_GPT\files_to_be_copied - Copy\IIIT-HW-Hindi_v1.tar\IIIT-HW-Hindi_v1\train.txt"
        try:
            with open(train_txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2 and parts[0] == image_key:
                        ground_truth = parts[1]
                        break
        except Exception as e:
            print(f"Could not read ground truth for {image_path}: {e}")
            
        new_row_df = pd.DataFrame([[ground_truth, result, image_path]], columns=results_df.columns)
        results_df = pd.concat([results_df, new_row_df], ignore_index=True)
        
        print(f"\nRecognized Text: '{result}'")
    
    #results_df.to_pickle("inference.pkl")
    results_df.to_csv("inference.csv")


def InferenceTrainData(model, tokenizer):
    pairs = GetPairsFromFile(r"C:\Users\kchauhan\Downloads\sahils\IIIT-HW-Hindi_v1\train.txt")
    pairs = pairs[:65000]#take only first 100
    InferenceOnPairs(model, tokenizer, pairs, "TrainData65k_krunal_epoch50_65kimages_VISUL_ENCODER_OFF_krunal_system_RTXPro6000")

def InferenceTestData(model, tokenizer):
    pairs = GetPairsFromFile(r"C:\Users\kchauhan\Downloads\sahils\IIIT-HW-Hindi_v1\test.txt")
    pairs = pairs[:12800]#take only first 100
    InferenceOnPairs(model, tokenizer, pairs, "TestData12800_krunal_epoch50_65kimages_VISUL_ENCODER_OFF_krunal_system_RTXPro6000")
def InferenceValtData(model, tokenizer):
    pairs = GetPairsFromFile(r"C:\Users\kchauhan\Downloads\sahils\IIIT-HW-Hindi_v1\val.txt")
    pairs = pairs[:12700]#take only first 100
    InferenceOnPairs(model, tokenizer, pairs, "ValData12700_krunal_epoch36_65kimages_VISUL_ENCODER_OFF_krunal_system_RTXPro6000")

#pairs: image path + ground truth
def InferenceOnPairs(model, tokenizer, pairs, file_name):
    results_df = pd.DataFrame(columns=['ground_truth', 'predicted_text', "Pass/Fail",'image_path'])

    for pair in pairs:
        test_image_path = pair[0]
        ground_truth = pair[1]
        if not os.path.exists(test_image_path):
            print("image does not exist")
            return
        
    
        print(f"Reading image: {test_image_path}")
        result = predict_text(model, tokenizer, test_image_path)
        print(f"\nRecognized Text: '{result}'")
    
        new_row_df = pd.DataFrame([[ground_truth, result, ground_truth == result, test_image_path]], columns=results_df.columns)
        results_df = pd.concat([results_df, new_row_df], ignore_index=True)
    
    results_df.to_pickle(file_name+"_inference.pkl")
    results_df.to_csv(file_name+"_inference.csv")


#filePath=r"C:\Users\j\Desktop\cs230\Patch_GPT\files_to_be_copied - Copy\IIIT-HW-Hindi_v1.tar\IIIT-HW-Hindi_v1\test.txt"
def GetPairsFromFile(filePath):
    test_txt_path = filePath
    prefix = r"C:\Users\kchauhan\Downloads\sahils\IIIT-HW-Hindi_v1\HindiSeg"
    path_text_pairs = []

    try:
        with open(test_txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    relative_path, hindi_text = parts
                    full_path = os.path.join(prefix, relative_path.replace('/', os.sep))
                    path_text_pairs.append((full_path, hindi_text))
    except Exception as e:
        print(f"Error reading or parsing test.txt: {e}")

    return path_text_pairs

if __name__ == "__main__":
    #do not train if save_path present
    #Train()
    if MODEL_MODE == "INFERENCE":
        Inference()
    elif MODEL_MODE == "TRAIN":
        Train()
    else:
        print("Invalid model mode")


