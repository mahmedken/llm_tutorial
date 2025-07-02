"""
this script finetunes a bert model for text classification on the imdb movie reviews dataset.
"""

# library imports
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from collections import Counter

# set device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using device: {device}")

class ImdbDataset(Dataset):
    """
    custom dataset class for imdb reviews
    
    this class handles the preprocessing of imdb text data for bert input.
    it tokenizes the text and converts it to the format expected by bert.
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        """        
        args:
            texts: list of review texts
            labels: list of sentiment labels (0=negative, 1=positive)
            tokenizer: bert tokenizer
            max_length: maximum sequence length for bert
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        get a single item from the dataset
        
        returns a dictionary with input_ids, attention_mask, and labels
        formatted for bert input
        """
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # tokenize the text using bert tokenizer
        # this converts text to tokens, then to input ids
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,      # add [cls] and [sep] tokens
            max_length=self.max_length,   # pad or truncate to max_length
            padding='max_length',         # pad shorter sequences
            truncation=True,              # truncate longer sequences
            return_attention_mask=True,   # return attention mask
            return_tensors='pt'           # return pytorch tensors
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_imdb_data(dataset_fraction=1.0):
    """
    load and prepare the imdb dataset
    
    the imdb dataset contains 50k movie reviews labeled as positive or negative.
    we'll split the original test set into validation (1/5) and test (4/5).
    """
    print("loading imdb dataset...")
    
    # load the dataset using huggingface datasets
    dataset = load_dataset("imdb")
    
    # extract train and original test data
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']
    
    # use only a fraction of the dataset if specified
    if dataset_fraction < 1.0:
        train_size = int(len(train_texts) * dataset_fraction)
        test_size = int(len(test_texts) * dataset_fraction)
        
        # create random indices for shuffling
        train_indices = torch.randperm(len(train_texts), generator=torch.Generator().manual_seed(420))[:train_size]
        test_indices = torch.randperm(len(test_texts), generator=torch.Generator().manual_seed(420))[:test_size]

        # sample using shuffled indices
        train_texts = [train_texts[i] for i in train_indices]
        train_labels = [train_labels[i] for i in train_indices]
        test_texts = [test_texts[i] for i in test_indices]
        test_labels = [test_labels[i] for i in test_indices]
        
        print(f"using {dataset_fraction:.1%} of dataset")

        print(f'train label distribution: {Counter(train_labels)}')
        print(f'test label distribution: {Counter(test_labels)}')
    
    # split original test set into validation (20%) and test (80%)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        test_texts, test_labels, 
        test_size=0.8, random_state=42, stratify=test_labels
    )
    
    print(f"train samples: {len(train_texts)}")
    print(f"validation samples: {len(val_texts)}")
    print(f"test samples: {len(test_texts)}")
    print(f"example review: {train_texts[0][:200]}...")
    print(f"example label: {train_labels[0]} ({'positive' if train_labels[0] == 1 else 'negative'})")
    
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels

def create_data_loaders(train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, 
                       tokenizer, batch_size=16, max_length=512):
    """
    create pytorch data loaders for training, validation, and testing
    
    data loaders handle batching and shuffling of the data during training.
    """
    print("creating data loaders...")
    
    # create dataset objects
    train_dataset = ImdbDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = ImdbDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = ImdbDataset(test_texts, test_labels, tokenizer, max_length)
    
    # create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,          # shuffle training data
        num_workers=2          # parallel data loading
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,         # shuffling is not needed for val data 
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,         # shuffling is not needed for test data
        num_workers=2
    )
    
    return train_loader, val_loader, test_loader

def initialize_model(num_classes=2):
    """
    initialize bert model for sequence classification
    
    we use bert-base-uncased which is a good starting point for most tasks.
    the model is pre-trained and we'll fine-tune it on our classification task.
    """
    print("initializing bert model...")
    
    # load pre-trained bert model with a classification head
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=num_classes,      # 2 for binary classification
        output_attentions=False,     # we don't need attention weights
        output_hidden_states=False   # we don't need hidden states
    )
    
    # move model to device (gpu if available)
    model.to(device)
    
    return model

def print_model_architecture(model):
    """
    print the model architecture for educational purposes
    """
    print("\nmodel architecture:")
    print("=" * 50)
    print(model)
    
    # count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\ntotal parameters: {total_params:,}")
    print(f"trainable parameters: {trainable_params:,}")
    print("=" * 50)

def train_model(model, train_loader, val_loader, epochs=3, learning_rate=2e-5, eval_steps=300):
    """
    train the bert model on the imdb dataset
    
    this function implements the main training loop with proper optimization
    and learning rate scheduling recommended for bert fine-tuning.
    """
    print("starting training...")
    
    # set up optimizer
    # adamw is recommended for transformer models
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    
    # calculate total training steps for lr scheduler
    total_steps = len(train_loader) * epochs
    
    # set up learning rate scheduler
    # linear warmup followed by linear decay is standard for bert
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,           # no warmup steps
        num_training_steps=total_steps
    )
    
    # training history
    history = {
        'step_numbers': [],
        'step_train_losses': [],
        'step_train_accuracies': [],
        'step_val_losses': [],
        'step_val_accuracies': [],
        'epoch_train_losses': [],
        'epoch_train_accuracies': [],
        'epoch_val_losses': [],
        'epoch_val_accuracies': []
    }
    
    for epoch in range(epochs):
        print(f"\nepoch {epoch + 1}/{epochs}")
        print("-" * 50)
        
        # training phase
        model.train()
        total_train_loss = 0
        train_predictions = []
        train_true_labels = []
        
        # training loop with progress bar
        train_progress = tqdm(train_loader, desc="training")
        global_step = epoch * len(train_loader)
        
        for batch_idx, batch in enumerate(train_progress):
            # move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # clear gradients
            optimizer.zero_grad()
            
            # forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            # backward pass
            loss.backward()
            
            # clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # update weights
            optimizer.step()
            scheduler.step()
            
            # accumulate loss and predictions
            total_train_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            train_predictions.extend(predictions.cpu().numpy())
            train_true_labels.extend(labels.cpu().numpy())
            
            # update progress bar
            train_progress.set_postfix({'loss': loss.item()})

            # evaluate both train and val metrics every n steps
            current_step = global_step + batch_idx + 1
            if current_step % eval_steps == 0:
                # calculate current training metrics
                current_train_acc = accuracy_score(train_true_labels, train_predictions)
                current_train_loss = total_train_loss / (batch_idx + 1)
                
                # calculate validation metrics
                val_accuracy, val_loss = evaluate_model(model, val_loader)
                
                # store step-wise metrics
                history['step_numbers'].append(current_step)
                history['step_train_losses'].append(current_train_loss)
                history['step_train_accuracies'].append(current_train_acc)
                history['step_val_losses'].append(val_loss)
                history['step_val_accuracies'].append(val_accuracy)
                
                print(f"\nstep {current_step} - train: loss={current_train_loss:.4f}, acc={current_train_acc:.4f}")
                print(f"step {current_step} - val: loss={val_loss:.4f}, acc={val_accuracy:.4f}")
                model.train()  # switch back to training mode
        
        # calculate epoch-end metrics
        epoch_train_loss = total_train_loss / len(train_loader)
        epoch_train_accuracy = accuracy_score(train_true_labels, train_predictions)
        epoch_val_accuracy, epoch_val_loss = evaluate_model(model, val_loader)
        
        # store epoch-wise metrics
        history['epoch_train_losses'].append(epoch_train_loss)
        history['epoch_train_accuracies'].append(epoch_train_accuracy)
        history['epoch_val_losses'].append(epoch_val_loss)
        history['epoch_val_accuracies'].append(epoch_val_accuracy)
        
        print(f"epoch {epoch + 1} - train: loss={epoch_train_loss:.4f}, acc={epoch_train_accuracy:.4f}")
        print(f"epoch {epoch + 1} - val: loss={epoch_val_loss:.4f}, acc={epoch_val_accuracy:.4f}")
    
    # plot training history
    plot_training_history(history)
    
    return history

def evaluate_model(model, data_loader, return_detailed=False):
    """
    evaluate the model on given data
    
    this function runs the model in evaluation mode (no gradient computation)
    and calculates accuracy and loss.
    """
    model.eval()
    predictions = []
    true_labels = []
    total_loss = 0
    
    with torch.no_grad():  # disable gradient computation for efficiency
        for batch in tqdm(data_loader, desc="evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            batch_predictions = torch.argmax(logits, dim=-1)
            
            total_loss += loss.item()
            predictions.extend(batch_predictions.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    avg_loss = total_loss / len(data_loader)
    
    if return_detailed:
        # return detailed metrics for final evaluation
        report = classification_report(
            true_labels, 
            predictions, 
            target_names=['negative', 'positive'],
            digits=4
        )
        print(f"\nfinal test accuracy: {accuracy:.4f}")
        print(f"average test loss: {avg_loss:.4f}")
        print("\nclassification report:")
        print(report)
        return accuracy, avg_loss, predictions, true_labels
    
    return accuracy, avg_loss

def plot_training_history(history):
    """
    plot training and validation losses and accuracies in wandb style
    """
    # set wandb-style colors and styling
    plt.style.use('default')
    colors = {'train': '#1f77b4', 'val': '#ff7f0e'}  # blue and orange
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # plot losses
    ax1.plot(history['step_numbers'], history['step_train_losses'], color=colors['train'], linewidth=2, 
             label='train', alpha=0.9)
    ax1.plot(history['step_numbers'], history['step_val_losses'], color=colors['val'], linewidth=2, 
             label='val', alpha=0.9)
    ax1.set_title('loss', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('steps', fontsize=12)
    ax1.set_ylabel('loss', fontsize=12)
    ax1.legend(frameon=False, fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_facecolor('#fafafa')
    
    # plot accuracies
    ax2.plot(history['step_numbers'], history['step_train_accuracies'], color=colors['train'], linewidth=2,
             label='train', alpha=0.9)
    ax2.plot(history['step_numbers'], history['step_val_accuracies'], color=colors['val'], linewidth=2,
             label='val', alpha=0.9)
    ax2.set_title('accuracy', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('steps', fontsize=12)
    ax2.set_ylabel('accuracy', fontsize=12)
    ax2.legend(frameon=False, fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_facecolor('#fafafa')
    
    # adjust layout and styling
    fig.patch.set_facecolor('white')
    plt.tight_layout(pad=3.0)
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print("training history plot saved as 'training_history.png'")

def save_model(model, tokenizer, save_path="./bert_finetuned"):
    """
    save the fine-tuned model and tokenizer
    
    this allows you to load and use the model later without retraining.
    """
    print(f"saving model to {save_path}...")
    
    # create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # save model and tokenizer
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print("model saved successfully!")

def load_model(model_path="./bert_finetuned"):
    """
    load a saved model and tokenizer
    
    use this to load your fine-tuned model for inference.
    """
    print(f"loading model from {model_path}...")
    
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    
    print("model loaded successfully!")
    return model, tokenizer

def predict_sentiment(text, model, tokenizer, max_length=512):
    """
    predict sentiment for a single text
    
    this function demonstrates how to use the fine-tuned model for inference
    on new, unseen text.
    """
    model.eval()
    
    # tokenize the input text (modern approach)
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # make prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        prediction = torch.argmax(logits, dim=-1).item()
    
    # convert to human readable format
    sentiment = "positive" if prediction == 1 else "negative"
    confidence = probabilities[0][prediction].item()
    
    return sentiment, confidence

def main():
    """
    main function that orchestrates the entire training process
    
    this is the entry point for the tutorial. it demonstrates the complete
    workflow from data loading to model evaluation.
    """
    print("bert fine-tuning tutorial for text classification")
    print("=" * 60)
    
    # hyperparameters
    batch_size = 16          # adjust based on your gpu memory
    max_length = 512         # maximum input sequence length
    epochs = 3               # number of training epochs
    learning_rate = 2e-5     # learning rate for fine-tuning
    dataset_fraction = 0.1   # use 10% of dataset for faster training (set to 1.0 for full dataset)
    eval_steps = 50         # evaluate every n steps
    
    # step 1: load data
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_imdb_data(dataset_fraction)
    
    # step 2: initialize tokenizer
    print("\nloading bert tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # step 3: create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels,
        tokenizer, batch_size, max_length
    )
    
    # step 4: initialize model
    model = initialize_model(num_classes=2)
    
    print_model_architecture(model)
    
    # step 5: train the model
    training_history = train_model(
        model, train_loader, val_loader, epochs, learning_rate, eval_steps
    )
    
    # step 6: detailed evaluation on test set
    print("\nperforming detailed evaluation on test set...")
    final_accuracy, final_loss, predictions, true_labels = evaluate_model(
        model, test_loader, return_detailed=True
    )
        
    print(f"\ntraining completed!")
    print(f"final test accuracy: {final_accuracy:.4f}")
    print(f"final test loss: {final_loss:.4f}")

    # step 7: save the model
    save_model(model, tokenizer)


if __name__ == "__main__":
    main()
