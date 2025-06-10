import torch
from torch import nn
import math

class InputEmbedding(nn.Module):
  """ Takes in input and gives vector of size d_model """
  def __init__(self, d_model : int, vocab_size:int):
    super().__init__()
    self.d_model = d_model #512
    self.vocab_size = vocab_size

    self.embedding = nn.Embedding(vocab_size, d_model)
  def forward(self, x : torch.tensor): # x are the token id, embeddings are scaled by d_model sqrt
        """Example token_id [1,81,57,654,78,2]"""
        return self.embedding(x.long()) * math.sqrt(self.d_model) # Ensure x is long for embedding


class PositionalEmbedding(nn.Module):
  """Deterministic approach to generation of positional embedding"""

  def __init__(self, d_model : int, seq_len : int, dropout: float) -> None:

    """
        Idea : Generated positional embeddings (should be same as that of
                   d_model) are to be added to word embeddings before sending
                   them to MultiHeadAttention Block.

                   Might get confusing on how can we combine information just by
                   a simple addition without the information cancelling each
                   other something that we see in signals.

                   Turns out the both the information don't cancel out each
                   other.

        Why dropout : Dropout prevents the model from learning too specific
                      pattern instead of learning or generalizing the problem.

                      ex : Training set : | Didn't enjoy the food | -ve
                                          | Loved the show        | +ve
                                          | Hated the whole movie | -ve
                                          | Pointless theory      | -ve
                                          | _____ ..........      | ____

                            Model might assume or give high weight to the first
                            position to get the overal sentiment of the
                            sentence.
    """

    super().__init__()

    self.d_model = d_model # 512
    self.seq_len = seq_len # Maximum Sequence Length
    self.dropout = nn.Dropout(dropout)

    # Create a matrix of shape (seq_ln, d_model) containing positional embeddings
    pe = torch.zeros(seq_len, d_model)

    # Position Vector [ NUMERATOR ]
    # Simple Indexing does'nt work out for many reasons
    position = torch.arange(0,seq_len, dtype=torch.float).unsqueeze(1) # Starts from zero not 1

    # Denominator [Updated for numerical stability]
    div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0) / d_model))

    # Even pos - sine and Odd pos - cosine
    pe[:,0::2] = torch.sin(position * div_term)
    pe[:,1::2] = torch.cos(position * div_term)

    # Batchify
    pe = pe.unsqueeze(0) # (1, seq_len, d_model)

    # Save the positional embedding as model's state dict by running following command
    # Since this is a determinsitic approach we can also not do this and create our own pe which is time taking and must be equal to the ones during inferencing
    self.register_buffer('pe',pe)

  def forward(self,x):
    """ Here x is the input embedding matrix """

    x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # Matrix Addition // Operator Overloading concept |||| x is in batches
    return self.dropout(x)


class LayerNormalization(nn.Module):

  def __init__(self, eps : float = 10**-6) -> None:
    """
        Normalizes individual items in batches independent of other items also
        uses two parameters alpha and bias that does amplification of normalised
        ouputs.

        The dimensions doesn't get changed due to normalization
    """
    super().__init__()
    self.eps = eps # Numerical Stability
    self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
    self.bias = nn.Parameter(torch.zeros(1)) # Added

  def forward(self,x):
    mean = x.mean(dim = -1, keepdim = True) # By default x.mean cancels the dimension information on which it was applied
    std = x.std(dim = -1, keepdim = True)
    return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):

  def __init__(self, d_model : int, d_ff : int, dropout : float) -> None:

    """
      Contains two linear layers
    """

    super().__init__()
    self.linear_1 = nn.Linear(d_model, d_ff) # W1 B1 ,, d_ff = 2048 paper
    self.dropout = nn.Dropout(dropout) # Output -> Activations -> Dropout -> Input
    self.linear_2 = nn.Linear(d_ff, d_model) # W2 B2

  def forward(self, x):
    """ Forward Block will receive the concatenated outputs of multihead attetion """

    # [batch_size | sequence_length | d_model ] -> [batch_size | sequence_length | d_ff] -> [batch_size | sequence_length | d_model]
    return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):

  def __init__(self, d_model:int, h:int, dropout:float) -> None:
    """Dynamically calulates the sizes and other information based on h"""

    super().__init__()

    self.d_model = d_model
    self.h = h # num_heads in MultiHead Attention
    assert d_model % h == 0, " d_model not divisible by h"

    self.d_k = d_model // h # 512 // 8 -> Each 64

    self.w_q = nn.Linear(d_model,d_model) # Linear layer not raw weight
    self.w_k = nn.Linear(d_model,d_model)
    self.w_v = nn.Linear(d_model,d_model)

    self.w_o = nn.Linear(d_model, d_model)
    self.dropout = nn.Dropout(dropout)

  @staticmethod
  def attention(query, key, value, mask, dropout: nn.Dropout):
        """
        1. Calulate attention scores by doing dot product of each query to all key
           vectors
        2. Rescale it by sqrt(d_k) as proposed by original tranformer paper
        3. Apply masking to remove unecesary interactions
        4. Apply dropout
        5. Return the values by weighting using attention scores
        """
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores



  def forward(self,q,k,v,mask):
    """
    0. Later we are going to see that q,k,v are equal to x
       [word embedding + postional embedding]

    1. Compute query key value matrix

    2. Reshape the query into a format such that each attention head of MHA
       has access to all words (sequence_length dim) but only sees d_model // h
       dimensions of d_model

       ex : Original dimension - 512, each 8 heads will see 512 // 8 = 64 dims

    3. Apply attention and concatenate individual heads' output to get final
       concatenated output

    4. Pass the final concatenated output through a linear layer and return the
       the result

    ##
      NOTE : Usage of `multiple heads` nomenclature might confuse some that we
             are using multiple neural networks and to each neural network we
             pass the separated query key value but that doesn't happen in
             praticality.

             We reshape and transpose the query key matrix in such a way that
             it seems
    ##

    """
    # 1. [batch_sequence | sequence_length | d_model] -> [batch_size | sequence_length | d_model] for all three transformations
    query = self.w_q(q)
    key = self.w_k(k)
    value = self.w_v(v)

    # 2. Dividing above matrices to a format accesible by individual heads
    # [batch_size | sequence_length | d_model] -> [batch_size | sequence_length | h | d_k] -> [batch_size | h | sequence_length | d_k ]
    query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
    key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
    value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

    x, self.attenttion_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

    # 3. Kind of reverse process that we did to divide the query
    # [batch_size | h | sequence_length | d_k] -> [batch_size | sequence_length | h | d_k] -> [batch_size | sequence_length | d_model]
    x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h * self.d_k)

    # 4. [batch_size | sequence_length | d_model] -> [batch_size | sequence_length | d_model]
    return self.w_o(x)

class ResidualConnection(nn.Module):

  def __init__(self, dropout : float) -> None:
    """
    1. Residual Connection are helps the gradient to flow properly in very deep
       networks

    2. Present in ResNet U-Net ViTs etc
    """
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.norm = LayerNormalization()

  def forward(self, x, sublayer): # Sublayer -> Previous layer
    return x + self.dropout(sublayer(self.norm(x)))
class Encoder(nn.Module):
  """ Multiple Encoder Block """

  def __init__(self, layers:nn.ModuleList) -> None:
    super().__init__()
    self.layers = layers
    self.norm = LayerNormalization()

  def forward(self, x, mask):
    for layer in self.layers:
      x = layer(x,mask) # Output of one encoder flows to another one. Since encoder doesn;t change dimension we can reuse the template

    return self.norm(x)


class EncoderBlock(nn.Module):
  """ Transformers has multiple encoder blocks. This is the template for one Encoder block"""

  def __init__(self, self_attention_block : MultiHeadAttentionBlock, feed_forward_block : FeedForwardBlock, dropout : float) -> None:
    super().__init__()
    self.self_attention_block = self_attention_block
    self.feed_forward_block = feed_forward_block
    self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)]) # Container for sub-modules doesnot define how the modules are connected like in nn.Sequential

  def forward(self, x, src_mask):
    """
    Defined intems of residual connections
    """
    # src_mask -> mask to mask the input of encoder
    # lambda ensures that we don't calulate it here rather we calulate it in residual connection defined above
    x = self.residual_connections[0](x, lambda x:self.self_attention_block(x,x,x,src_mask))
    x = self.residual_connections[1](x, self.feed_forward_block)

    return x

class DecoderBlock(nn.Module):
  """Has cross attention"""

  def __init__(self,self_attention_block : MultiHeadAttentionBlock, cross_attention_block : MultiHeadAttentionBlock, feed_forward_block : FeedForwardBlock, dropout : float):
    super().__init__()

    self.self_attention_block = self_attention_block
    self.cross_attention_block = cross_attention_block
    self.feed_forward_block = feed_forward_block
    self.residual_connections = nn.ModuleList([ResidualConnection(dropout=dropout) for _ in range(3)])

  def forward(self, x, encoder_output, src_mask, tgt_mask):
    # 2 masks one for encoder and one for decoder
    x = self.residual_connections[0](x, lambda x : self.self_attention_block(x,x,x,tgt_mask)) # Query Key Value
    x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x , encoder_output,encoder_output,src_mask))
    x = self.residual_connections[2](x, self.feed_forward_block)

    return x

class Decoder(nn.Module):
  """ Contains multiple decoder block """

  def __init__(self,layers:nn.ModuleList):
    super().__init__()
    self.layers = layers
    self.norm = LayerNormalization()

  def forward(self, x, encoder_output, src_mask, tgt_mask):
    for layer in self.layers:
      x = layer(x,encoder_output, src_mask, tgt_mask)
    return self.norm(x)

class ProjectionLayer(nn.Module):

  def __init__(self,d_model,vocab_size):
    super().__init__()
    self.proj = nn.Linear(d_model,vocab_size)

  def forward(self, x):
    # BS SL d_model -----> BS SL VS
    return torch.log_softmax(self.proj(x), dim = -1) # Probablities

class Transformer(nn.Module):

  def __init__(self, encoder : Encoder, decoder : Decoder, src_embed: InputEmbedding, tgt_embed : InputEmbedding, src_pos : PositionalEmbedding, tgt_pos : PositionalEmbedding, projection_layer:ProjectionLayer):
    super().__init__()

    self.encoder_layer = encoder
    self.decoder_layer = decoder
    self.src_embed = src_embed
    self.tgt_embed = tgt_embed
    self.src_pos = src_pos
    self.tgt_pos = tgt_pos
    self.projection_layer = projection_layer

  def encoder(self, src, src_mask):
    src = self.src_embed(src)
    src = self.src_pos(src)
    return self.encoder_layer(src, src_mask)

  def decoder(self, encoder_output, src_mask, tgt, tgt_mask):
    tgt = self.tgt_embed(tgt)
    tgt = self.tgt_pos(tgt)
    return self.decoder_layer(tgt, encoder_output, src_mask, tgt_mask)

  def project(self,x):
    return self.projection_layer(x)

def build_transformer(src_vocab_size : int, tgt_vocab_size : int, src_seq_len : int, tgt_seq_len : int, d_model = 512, N : int = 6, h : int = 8, dropout :float = 0.1, d_ff : int = 2048):
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    # Create pos encoding layers
    src_pos = PositionalEmbedding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEmbedding(d_model, tgt_seq_len, dropout)

    # Encoder Blocks
    encoder_blocks = []
    for _ in range(N):
      encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
      feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
      # Encoder Blocks takes in layers
      # And has operation defined
      # Remember encoder block is like a template
      encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout) # Kinda lika modulelist but with operations defined
      encoder_blocks.append(encoder_block)

    # Create decoder block
    decoder_blocks = []
    for _ in range(N):
      decoder_self_attention_block = MultiHeadAttentionBlock(d_model,h,dropout)
      decoder_cross_attention_block = MultiHeadAttentionBlock(d_model,h,dropout)
      feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
      decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout) # Kinda like modulelist but with operations defined
      decoder_blocks.append(decoder_block)

    # We then pass the series of encoder blocks for their interconnection defined in Encoder
    encoder = Encoder(nn.ModuleList(encoder_blocks)) # Now each encoder e
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # OKAY WE ARE THERE
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Intitialiseparameter smart Xavier

    for p in transformer.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)

    return transformer

