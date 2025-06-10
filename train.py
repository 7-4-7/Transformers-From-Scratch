# Tokenizer splits the senteces to tokens using many available techinques
# Word Level - split words by space
# After splitting tokenizer creates vocabulary [mapping between words and token_id]

import torch
import torch.nn as nn

from dataset import BilingualDataset, causal_mask
from model import build_transformer

from config import get_config, get_weights_file_path

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.data import DataLoader, Dataset, random_split

from pathlib import Path
import warnings


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    # Tokenizer_file - path to tokenizer file
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    
    if not Path.exists(tokenizer_path):
        # Create if not exist
        tokenizer: Tokenizer    = Tokenizer(WordLevel(unk_token = "[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ["[UNK]",
                                                     "[PAD]",
                                                     "[SOS]",
                                                     "[EOS]"],
                                   min_frequency = 2)
        
        tokenizer.train_from_iterator(get_all_sentences(ds,lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
        
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer

def get_ds(config):
    """Loading the dataset"""
    
    ds_raw = load_dataset('opus_books',f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')
    
    # Build tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
    
    # Splitting the data
    # Data is originally train split from hugging face
    # We manually split here
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'],config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'],config['lang_tgt'], config['seq_len'])
    
    max_len_src = 0
    max_len_tgt = 0
    
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_src.encode(item['translation'][config['lang_tgt']]).ids
        
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
        
    print(f"Max length of sequence src : {max_len_src}")
    print(f"Max length of sequence tgt : {max_len_tgt}")
    
    train_dataloader = DataLoader(train_ds, 
                                  batch_size = config['batch_size'],
                                  shuffle=True)
    
    val_dataloader = DataLoader(val_ds,
                                batch_size=1,
                                shuffle = True)
    
    return train_dataloader, val_dataloader, tokenizer_src,tokenizer_tgt

def get_model(config,vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'],config['seq_len'], config['d_model'])
    return model

def train_model(config):
    
    device = 'cuda'
    print(f"Using device : cuda")
    
    Path(config['model_folder']).mkdir(parents = True,exist_ok=True)
    train_dataloader, val_dataloader, tokenizer_src,tokenizer_tgt = get_ds(config)
    
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to('cuda')
    
    writer = SummaryWriter(config['experiment_name'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps = 1e-9)
    
    initial_epoch = 0
    global_step = 0
    
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        if not Path(model_filename).exists():
            print(f"Preload file {model_filename} does not exist. Skipping preload.")
        else:
            print(f"Preloading model {model_filename}")
            state = torch.load(model_filename)
            optimizer.load_state_dict(state['optimizer_state_dict'])
            global_step = state['global_step']
        
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'),
                                  label_smoothing = 0.1).to('cuda')
    
    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc = f'Processing Epoch{epoch:02d}')
        
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to('cuda')
            decoder_input = batch['decoder_input'].to('cuda')
            
            encoder_mask = batch['encoder_mask'].to('cuda')
            decoder_mask = batch['decoder_mask'].to('cuda')
            
            encoder_output = model.encoder(encoder_input, encoder_mask)
            decoder_output = model.decoder(encoder_output, encoder_mask, decoder_input, decoder_mask)
            
            proj_ouput = model.project(decoder_output)
            
            label = batch['label'].to('cuda')
            loss = loss_fn(proj_ouput.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            
            batch_iterator.set_postfix(loss=f"{loss.item():6.3f}")
            writer.add_scalar("train_loss",loss.item(), global_step)
            writer.flush()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save(
            {
                "epoch" : epoch,
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'global_step' : global_step,
            },model_filename
        )
        
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)









