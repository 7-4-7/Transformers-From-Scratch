import torch
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    
    def __init__(self, 
                 ds, 
                 tokenizer_src,
                 tokenizer_tgt,
                 src_lang,
                 tgt_lang,
                 seq_len):
        
        super().__init__()
        
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        
        # int64 = long
        # Get sos token id as tensor
        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype = torch.int64)
        
        # Get EOS token id as tensor
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype = torch.int64)
        
        # Get PAD token id as tensor
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype = torch.int64)
        
        
    def __len__(self):
        return len(self.ds)
        
    def __getitem__(self, index):
        
        # Extract original pair from hf
        src_targer_pair = self.ds[index]
        src_text = src_targer_pair['translation'][self.src_lang]
        tgt_text = src_targer_pair['translation'][self.tgt_lang]
        
        # Convert each pair of text to token id using HF tokenizer
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        
        # Add padding token
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # Only add the sos token to decoder input

        assert enc_num_padding_tokens >= 0, "Sequence Invalid"
        assert dec_num_padding_tokens >= 0, "Sequence Invalid"
        
        # Add SOS CONTENT EOS PADDING
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype = torch.int64)
            ]
        )
        
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64)
            ]
        )
        
        # EOS added (Expectation from decoder)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64)
            ]
        )
        
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        
        return {
            "encoder_input" : encoder_input,
            "decoder_input" : decoder_input,
            "encoder_mask"  : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # 1 1 Seq_len
            "decoder_mask"  : (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label"         : label,
            "src_text"      : src_text,
            "tgt_text"      : tgt_text,
            }
    
def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0
