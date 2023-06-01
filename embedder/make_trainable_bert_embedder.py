#%%
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer, BertModel


class BERTQuestionEmbedding(nn.Module):
    """BERT Question Embedding"""
    def __init__(self, bert_model_name, dropout, cat=False):
        super(BERTQuestionEmbedding, self).__init__()
        self.emb_dim = 768 # BERT default

        tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        model = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = tokenizer
        self.model = model

        self.vocab_size = self.model.embeddings.word_embeddings.weight.size()[0]
        self.padding_idx = self.find_padding_idx()
        self.emb = nn.Embedding(self.vocab_size+1, self.emb_dim,
                                padding_idx=self.padding_idx)
        self.cat = cat
        if cat:
            self.emb_ = nn.Embedding(self.vocab_size+1, self.emb_dim,
                                     padding_idx=self.padding_idx)
            self.emb_.weight.requires_grad = False # fixed

        self.dropout = nn.Dropout(dropout)


    def find_padding_idx(self):
        pad_id = self.tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
        return pad_id

    def init_bert_embedding(self):
        embedding_mat = self.model.embeddings.word_embeddings.weight.data
        self.emb.weight.data[:self.ntoken] = embedding_mat


    def forward(self, x: str):
        tokenized_text = self.tokenizer.tokenize(x)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(indexed_tokens)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        encoded_layers, _ = self.model(tokens_tensor, segments_tensors)
        token_embeddings = torch.stack(encoded_layers, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)

        token_vecs_sum = []
        for token in token_embeddings:
            sum_vec = torch.sum(token[-4:], dim=0)
            token_vecs_sum.append(sum_vec)

        return token_vecs_sum



# Example usage
bert_model_name = 'bert-base-uncased'
dropout = 0.1
model = BERTQuestionEmbedding(bert_model_name, dropout)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
#%%
# Update the embedding
EPOCHS = 10
epoch_embeds_1 = []
for epoch in range(EPOCHS):
    # Forward pass
    inputs = "Is this an axial plane?"
    embeds = model(inputs)
    # get first token embedding of each sentence
    print(embeds[0].shape)
    epoch_embeds_1.append(embeds[0])
    # Make a pseudo loss here
    loss = torch.tensor(0.15, requires_grad=True)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # Backward pass and update embeddings

# %%
# check if the embeddings are updated
for i in range(EPOCHS):
    print(torch.equal(epoch_embeds_1[i], epoch_embeds_1[1]))
# Trained
