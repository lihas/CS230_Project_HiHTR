"""
Minimal CRNN Training Example
Input: Image → Output: Text
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pdb
import torchvision.transforms as transforms
import collections
import unicodedata
import re

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)
        self.dropout = nn.Dropout2d(p=0.2)

    def forward(self, input):
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        t_rec = self.dropout(t_rec)

        output = self.embedding(t_rec)
        output = output.view(T, b, -1)

        return output


class ResidualBlock(nn.Module):
    def __init__(self, nIn1, nOut1, ks1, ss1, ps1,
                 nIn2, nOut2, ks2, ss2, ps2, downsample=None,
                 bn2_active=True, att_type=None, stn_inputsize=None,
                 stn_nheads=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(nIn1, nOut1, ks1, ss1, ps1)
        self.bn1 = nn.BatchNorm2d(nIn1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(nIn2, nOut2, ks2, ss2, ps2)
        self.downsample = downsample
        self.bn2_active = bn2_active
        if self.bn2_active:
            self.bn2 = nn.BatchNorm2d(nIn2)
        self.stn = None

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        if self.bn2_active:
            out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual

        if self.stn:
            rectified, _  = self.stn(out)
            out = rectified
        return out


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False, STN=True, att_type=None):
        super(CRNN, self).__init__()
        ks = [3, 3, 3, 3, 3,
              3, 3, 3, 3,
              3, 3, 3, 3,
              3, 3, 3, 3, 3]  # filter size
        ps = [1, 1, 1, 1, 1,
              1, 1, 1, 1,
              1, 1, 1, 1,
              1, 1, 1, 1, 0]  # padding
        ks = [3, 3, 3, 3, 3,
              3, 3, 3, 3,
              3, 3, 3, 3,
              3, 3, 3, 3, 3]  # filter size
        ps = [1, 1, 1, 1, 1,
              1, 1, 1, 1,
              1, 1, 1, 1,
              1, 1, 1, 1, 0]  # padding
        ss = [1, 1, 1, 1, 1,
              1, 1, 1, 1,
              1, 1, 1, 1,
              1, 1, 1, 1, 1]  # stride parameter
        nm = [64, 64, 64, 64, 64,
              128, 128, 128, 128,
              256, 256, 256, 256,
              512, 512, 512, 512, 512]  # number of channels

        cnn = nn.Sequential()

        def convRelu(i, BatchNorm2d=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if BatchNorm2d:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0, True)
        cnn.add_module('pooling{0}'.format(
            0), nn.MaxPool2d((2, 2), stride=(2, 2)))

        nIn1, nOut1, ks1, ss1, ps1 = nm[0], nm[1], ks[1], ss[1], ps[1]
        nIn2, nOut2, ks2, ss2, ps2 = nm[1], nm[2], ks[2], ss[2], ps[2]
        cnn.add_module('residualBlock{0}'.format(12), ResidualBlock(nIn1, nOut1, ks1, ss1, ps1,
                                                                    nIn2, nOut2, ks2, ss2, ps2,
                                                                    ))

        nIn1, nOut1, ks1, ss1, ps1 = nm[2], nm[3], ks[3], ss[3], ps[3]
        nIn2, nOut2, ks2, ss2, ps2 = nm[3], nm[4], ks[4], ss[4], ps[4]
        cnn.add_module('residualBlock{0}'.format(34), ResidualBlock(nIn1, nOut1, ks1, ss1, ps1,
                                                                    nIn2, nOut2, ks2, ss2, ps2,
                                                                    ))
        cnn.add_module('pooling{0}'.format(
            1), nn.MaxPool2d((2, 2), stride=(2, 2)))

        nIn1, nOut1, ks1, ss1, ps1 = nm[4], nm[5], ks[5], ss[5], ps[5]
        nIn2, nOut2, ks2, ss2, ps2 = nm[5], nm[6], ks[6], ss[6], ps[6]
        downsample = nn.Sequential(nn.Conv2d(nIn1, nOut2, 1, 1, 0))
        cnn.add_module('residualBlock{0}'.format(56), ResidualBlock(nIn1, nOut1, ks1, ss1, ps1,
                                                                    nIn2, nOut2, ks2, ss2, ps2,
                                                                    downsample,
                                                                    ))
        nIn1, nOut1, ks1, ss1, ps1 = nm[6], nm[7], ks[7], ss[7], ps[7]
        nIn2, nOut2, ks2, ss2, ps2 = nm[7], nm[8], ks[8], ss[8], ps[8]
        cnn.add_module('residualBlock{0}'.format(78), ResidualBlock(nIn1, nOut1, ks1, ss1, ps1,
                                                                    nIn2, nOut2, ks2, ss2, ps2,
                                                                    ))
        cnn.add_module('pooling{0}'.format(
            2), nn.MaxPool2d((2, 2), stride=(2, 2)))

        nIn1, nOut1, ks1, ss1, ps1 = nm[8], nm[9], ks[9], ss[9], ps[9]
        nIn2, nOut2, ks2, ss2, ps2 = nm[9], nm[10], ks[10], ss[10], ps[10]
        downsample = nn.Sequential(nn.Conv2d(nIn1, nOut2, 1, 1, 0))
        cnn.add_module('residualBlock{0}'.format(910), ResidualBlock(nIn1, nOut1, ks1, ss1, ps1,
                                                                     nIn2, nOut2, ks2, ss2, ps2,
                                                                     downsample,
                                                                    ))
        nIn1, nOut1, ks1, ss1, ps1 = nm[10], nm[11], ks[11], ss[11], ps[11]
        nIn2, nOut2, ks2, ss2, ps2 = nm[11], nm[12], ks[12], ss[12], ps[12]
        cnn.add_module('residualBlock{0}'.format(1112), ResidualBlock(nIn1, nOut1, ks1, ss1, ps1,
                                                                      nIn2, nOut2, ks2, ss2, ps2,
                                                                    ))
        cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d(
            (2, 2), stride=(2, 1), padding=(0, 1)))

        nIn1, nOut1, ks1, ss1, ps1 = nm[12], nm[13], ks[13], ss[13], ps[13]
        nIn2, nOut2, ks2, ss2, ps2 = nm[13], nm[14], ks[14], ss[14], ps[14]
        downsample = nn.Sequential(nn.Conv2d(nIn1, nOut2, 1, 1, 0))
        cnn.add_module('residualBlock{0}'.format(1314), ResidualBlock(nIn1, nOut1, ks1, ss1, ps1,
                                                                      nIn2, nOut2, ks2, ss2, ps2,
                                                                      downsample,
                                                                      att_type=att_type))
        nIn1, nOut1, ks1, ss1, ps1 = nm[14], nm[15], ks[15], ss[15], ps[15]
        nIn2, nOut2, ks2, ss2, ps2 = nm[15], nm[16], ks[16], ss[16], ps[16]
        cnn.add_module('residualBlock{0}'.format(1516), ResidualBlock(nIn1, nOut1, ks1, ss1, ps1,
                                                                      nIn2, nOut2, ks2, ss2, ps2,
                                                                      att_type=att_type))
        cnn.add_module('pooling{0}'.format(4), nn.MaxPool2d(
            (2, 2), stride=(2, 1), padding=(0, 1)))

        convRelu(17, True)
        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, 256, 256),
            BidirectionalLSTM(256, 256, nclass))

    def forward(self, input):
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)

        # rnn features
        output = self.rnn(conv)

        return output



# Levenshtein distance for calculating edit distance
def levenshtein(seq1, seq2):
    """
    Calculate Levenshtein distance between two sequences
    Used for computing CER and WER
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y
    
    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix[x, y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1] + 1,
                    matrix[x, y-1] + 1
                )
    
    return matrix[size_x - 1, size_y - 1]


class LabelConverter(object):
    """Convert between text labels and text indices"""
    
    def __init__(self, alphabet, ctc_blank='<blank>', use_NFKD=False, add_blank=False):
        """
        Args:
            alphabet: string or list of characters
            ctc_blank: blank token for CTC
            use_NFKD: whether to use Unicode NFKD normalization
            add_blank: whether to add blank between repeated characters
        """
        self.alphabet = list(alphabet)
        self.ctc_blank = ctc_blank
        self.use_NFKD = use_NFKD
        self.add_blank = add_blank
        
        # Create char2id and id2char mappings (0 reserved for blank)
        self.char2id = {char: i+1 for i, char in enumerate(self.alphabet)}
        self.char2id[ctc_blank] = 0
        
        self.id2char = {i+1: char for i, char in enumerate(self.alphabet)}
        self.id2char[0] = ctc_blank
        
        self.nclasses = len(self.id2char)
    
    def encode(self, text):
        """
        Encode text to indices
        
        Args:
            text: str or list of str
            
        Returns:
            torch.IntTensor: encoded text indices
            torch.IntTensor: length of each text
        """
        if isinstance(text, str):
            if self.use_NFKD:
                text = unicodedata.normalize('NFKD', text)
            
            entext = []
            skipped_chars = []
            
            for i, char in enumerate(text):
                # Add blank between repeated characters if needed
                if i > 0 and char == text[i-1] and self.add_blank:
                    entext.append(self.char2id[self.ctc_blank])
                
                # Encode character if in vocabulary
                if char in self.char2id:
                    entext.append(self.char2id[char])
                else:
                    skipped_chars.append(char)
            
            # Report skipped characters
            if skipped_chars:
                unique_skipped = set(skipped_chars)
                print(f"Warning: Skipped {len(skipped_chars)} unknown characters: {unique_skipped}")
            
            length = [len(entext)]
            return entext, length
            
        elif isinstance(text, collections.abc.Iterable):
            # Batch encoding
            entext = []
            length = []
            for s in text:
                t, l = self.encode(s)
                entext += t
                length += l
            return torch.IntTensor(entext), torch.IntTensor(length)
        
        return torch.IntTensor([]), torch.IntTensor([0])
    
    def decode(self, indices, lengths, raw=False):
        """
        Decode indices back to text
        
        Args:
            indices: torch.IntTensor of character indices
            lengths: torch.IntTensor of lengths
            raw: if True, show all characters including blanks
            
        Returns:
            str or list of str: decoded text(s)
        """
        if lengths.numel() == 1:
            length = lengths[0]
            if raw:
                # Show all characters including blank
                char_list = []
                for i in range(length):
                    if indices[i] != 0:
                        char_list.append(self.id2char[indices[i].item()])
                    else:
                        char_list.append('~')  # blank placeholder
                return ''.join(char_list)
            else:
                # CTC decoding: remove blanks and repeated characters
                char_list = []
                for i in range(length):
                    if indices[i] != 0 and (not (i > 0 and indices[i - 1] == indices[i])):
                        char_list.append(self.id2char[indices[i].item()])
                return ''.join(char_list)
        else:
            # Batch decoding
            texts = []
            index = 0
            for i in range(lengths.numel()):
                l = lengths[i]
                texts.append(
                    self.decode(indices[index:index + l], torch.IntTensor([l]), raw=raw)
                )
                index += l
            return texts


# Dataset
class TextDataset(Dataset):
    def __init__(self, labels_file_name, relative_path_to_images, vocabulary=None, 
                 clean_labels=True, encoding='utf-8', idx_start = 0, idx_end = 5000):
        """
        Args:
            labels_file_name: path to file with image_name label pairs
            relative_path_to_images: path prefix for images
            vocabulary: list/string of valid characters (optional)
            clean_labels: whether to remove out-of-vocabulary characters
            encoding: file encoding
        """
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
        to_remove = re.search(out_of_vocab, label)
        if to_remove:
            pattern = re.compile(out_of_vocab)
            cleaned = pattern.sub('', label)
            return cleaned
        
        return label
    
    def __getitem__(self, idx):
        # Load & preprocess image
        try:
            img = cv2.imread(self.images[self.idx_start + idx])
            if img is None:
                print(f"Failed to load image: {self.images[idx]}")
                return None
            
            # Resize to (width=256, height=96)
            img = cv2.resize(img, (256, 96), interpolation=cv2.INTER_AREA)
            
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


# Training function
def train(model, train_loader, converter, epochs=10, lr=0.001, display_interval=10):
    """
    Train the model with proper label encoding
    
    Args:
        model: CRNN model
        train_loader: DataLoader
        converter: LabelConverter instance
        epochs: number of epochs
        lr: learning rate
        display_interval: how often to display loss
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Training on device: {device}")
    print(f"Number of classes: {converter.nclasses}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_idx, (images, label_texts) in enumerate(train_loader):
            if images is None or label_texts is None:
                continue
            
            # Encode labels using converter (similar to base_train.py)
            encoded_labels, label_lengths = converter.encode(label_texts)
            
            # Move to device
            images = images.to(device)
            encoded_labels = encoded_labels.to(device)
            label_lengths = label_lengths.to(device)
            
            # Forward pass
            outputs = model(images)  # (T, batch, num_classes)
            batch_size = images.size(0)
            
            # Input lengths for CTC (all sequences have same length from model)
            input_lengths = torch.full((batch_size,), outputs.size(0), dtype=torch.long)
            
            # Apply log_softmax for CTC loss
            log_probs = torch.nn.functional.log_softmax(outputs, dim=2)
            
            # Calculate CTC loss
            loss = criterion(log_probs, encoded_labels, input_lengths, label_lengths)
            
            # Check for NaN loss (similar to base_train.py)
            if torch.isnan(loss):
                print(f"NaN loss detected at batch {batch_idx}")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            # Display progress
            if (batch_idx + 1) % display_interval == 0:
                avg_loss = total_loss / batch_count
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {avg_loss:.4f}")
        
        # Epoch summary
        avg_epoch_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} Complete - Average Loss: {avg_epoch_loss:.4f}")
    
    return model


# Predict function
def predict(model, image_path, converter):
    """
    Predict text from image using the model and converter
    
    Args:
        model: trained CRNN model
        image_path: path to image file
        converter: LabelConverter instance for decoding
        
    Returns:
        str: predicted text
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Preprocess image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    # Resize to (width=256, height=96)
    img = cv2.resize(img, (256, 96), interpolation=cv2.INTER_AREA)
    
    # Convert to (batch, channels, height, width) format
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Normalize to [-1, 1]
    img = torch.from_numpy(img).float() / 255.0
    img = (img - 0.5) / 0.5
    img = img.to(device)
    
    # Predict
    with torch.no_grad():
        output = model(img)  # (T, batch, num_classes)
    
    # Decode using converter (similar to base_train.py)
    log_probs = torch.nn.functional.log_softmax(output, dim=2)
    
    # Get predictions: argmax across character dimension
    _, preds = log_probs[:, 0, :].max(1)  # (T,)
    preds_size = torch.IntTensor([preds.size(0)])
    
    # Decode predictions using converter
    decoded_text = converter.decode(preds, preds_size, raw=False)
    
    return decoded_text


# Evaluation function to calculate CER and WER
def evaluate(model, data_loader, converter, max_samples=None, show_examples=5):
    """
    Evaluate model on dataset and calculate CER/WER
    
    Args:
        model: trained CRNN model
        data_loader: DataLoader for evaluation data
        converter: LabelConverter instance
        max_samples: maximum number of samples to evaluate (None for all)
        show_examples: number of prediction examples to show
        
    Returns:
        dict with 'cer', 'wer', 'total_chars', 'total_words', 'correct_words'
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    total_char_errors = 0
    total_chars = 0
    total_word_errors = 0
    total_words = 0
    correct_words = 0
    
    predictions = []
    ground_truths = []
    
    print("\n" + "="*60)
    print("Evaluating Model")
    print("="*60)
    
    sample_count = 0
    with torch.no_grad():
        for batch_idx, (images, label_texts) in enumerate(data_loader):
            if images is None or label_texts is None:
                continue
            
            # Move to device
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)  # (T, batch, num_classes)
            log_probs = torch.nn.functional.log_softmax(outputs, dim=2)
            
            # Decode predictions for each image in batch
            batch_size = images.size(0)
            for i in range(batch_size):
                # Get prediction for single image
                _, preds = log_probs[:, i, :].max(1)  # (T,)
                preds_size = torch.IntTensor([preds.size(0)])
                
                # Decode prediction
                pred_text = converter.decode(preds, preds_size, raw=False)
                gt_text = label_texts[i]
                
                predictions.append(pred_text)
                ground_truths.append(gt_text)
                
                # Calculate character-level errors
                char_errors = levenshtein(gt_text, pred_text)
                total_char_errors += char_errors
                total_chars += len(gt_text)
                
                # Calculate word-level errors
                if gt_text != pred_text:
                    total_word_errors += 1
                else:
                    correct_words += 1
                total_words += 1
                
                sample_count += 1
                
                # Check if we've reached max_samples
                if max_samples and sample_count >= max_samples:
                    break
            
            if max_samples and sample_count >= max_samples:
                break
            
            # Progress indicator
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {sample_count} samples...", end='\r')
    
    # Calculate metrics
    cer = (total_char_errors / total_chars * 100) if total_chars > 0 else 0
    wer = (total_word_errors / total_words * 100) if total_words > 0 else 0
    accuracy = (correct_words / total_words * 100) if total_words > 0 else 0
    
    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Total Samples: {sample_count}")
    print(f"Character Error Rate (CER): {cer:.2f}%")
    print(f"Word Error Rate (WER): {wer:.2f}%")
    print(f"Word Accuracy: {accuracy:.2f}%")
    print(f"Total Characters: {total_chars}")
    print(f"Character Errors: {total_char_errors}")
    print(f"Total Words: {total_words}")
    print(f"Word Errors: {total_word_errors}")
    print(f"Correct Words: {correct_words}")
    
    # Show some examples
    if show_examples > 0:
        print("\n" + "="*60)
        print(f"Sample Predictions (first {show_examples}):")
        print("="*60)
        for i in range(min(show_examples, len(predictions))):
            print(f"\nSample {i+1}:")
            print(f"  Ground Truth: {ground_truths[i]}")
            print(f"  Prediction:   {predictions[i]}")
            if ground_truths[i] == predictions[i]:
                print(f"  ✓ CORRECT")
            else:
                print(f"  ✗ INCORRECT (edit distance: {levenshtein(ground_truths[i], predictions[i])})")
    
    return {
        'cer': cer,
        'wer': wer,
        'accuracy': accuracy,
        'total_chars': total_chars,
        'char_errors': total_char_errors,
        'total_words': total_words,
        'word_errors': total_word_errors,
        'correct_words': correct_words,
        'predictions': predictions,
        'ground_truths': ground_truths
    }


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


# Helper function to load alphabet from file
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


if __name__ == '__main__':
    model_save_path = 'training_results/trained_models/crnn_trained_hindi_50000.pth'
    lable_train_file = '../hindi_dataset/IIIT-HW-Hindi_v1/train.txt'
    training_idx_start = 0
    training_idx_end = training_idx_start + 65000
    
    testing_idx_start = training_idx_end
    testing_idx_end = testing_idx_start + 10
    
    train_epochs = 50
    # Load Hindi alphabet from file
    alphabet = load_alphabet('alphabet/hi.txt')
    
    converter = LabelConverter(
        alphabet=alphabet,
        ctc_blank='<blank>',
        use_NFKD=True,   # Set to True for Unicode normalization (good for Hindi)
        add_blank=False  # Set to True to add blanks between repeated chars
    )
    
    nclass = converter.nclasses
    print(f"Number of classes: {nclass}")
    print(f"Vocabulary size: {len(alphabet)}")
    
    # Create model
    model = CRNN(imgH=96, nc=3, nclass=nclass, nh=256, n_rnn=2)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
      
    
    # Create dataset
    dataset = TextDataset(
        labels_file_name=lable_train_file,
        relative_path_to_images='../hindi_dataset/IIIT-HW-Hindi_v1/',
        vocabulary=alphabet,  # Pass vocabulary for label cleaning
        clean_labels=True,    # Clean out-of-vocabulary characters
        encoding='utf-8',
        idx_start = training_idx_start,
        idx_end = training_idx_end
    )
    
    # Create DataLoader
    loader = DataLoader(
        dataset, 
        batch_size=64, 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to > 0 for parallel data loading
    )
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Number of batches: {len(loader)}")
    
    # Create validation dataset (optional - using subset of training data for demo)
    # In production, you should use a separate validation file
    val_dataset = TextDataset(
        labels_file_name=lable_train_file,  # Change to val.txt if available
        relative_path_to_images='../hindi_dataset/IIIT-HW-Hindi_v1/',
        vocabulary=alphabet,
        clean_labels=True,
        encoding='utf-8',
        idx_start = testing_idx_start,
        idx_end = testing_idx_end
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,  # Can use larger batch for evaluation
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Train
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    model = train(model, loader, converter, epochs=train_epochs, lr=0.001, display_interval=10)
    
    # Save model
    torch.save(model.state_dict(), model_save_path)
    print("\nModel saved to: crnn_trained_hindi.pth")
    
    # Evaluate on validation set
    print("\n" + "="*60)
    print("Evaluating on Validation Set")
    print("="*60)
    eval_results = evaluate(
        model=model,
        data_loader=val_loader,
        converter=converter,
        max_samples=None,  # Evaluate on 500 samples (set to None for all)
        show_examples=10   # Show 10 prediction examples
    )
    
    # Print summary
    print("\n" + "="*60)
    print("Final Metrics Summary")
    print("="*60)
    print(f"CER: {eval_results['cer']:.2f}%")
    print(f"WER: {eval_results['wer']:.2f}%")
    print(f"Word Accuracy: {eval_results['accuracy']:.2f}%")
    
    # Single image prediction test
    print("\n" + "="*60)
    print("Testing Single Image Prediction")
    print("="*60)
    test_image = '../hindi_dataset/IIIT-HW-Hindi_v1/HindiSeg/train/1/104/1.jpg'
    text = predict(model, test_image, converter)
    print(f"Predicted text: {text}")
    
    print("\nTraining complete!")

