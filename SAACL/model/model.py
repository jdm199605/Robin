import torch
import torch.nn as nn
import transformers
from transformers import BertTokenizer, BertModel

class SelfAttention(torch.nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert(
            self.head_dim * heads == embed_size
            )
        
        self.values = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)
        
        self.fc_out = torch.nn.Linear(heads * self.head_dim, embed_size)
    
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        values = values.view(N, value_len, self.heads, self.head_dim)
        keys = values.view(N, key_len, self.heads, self.head_dim)
        query = query.view(N, query_len, self.heads, self.head_dim)
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)
        
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
            
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
            
        attention = torch.nn.functional.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        # Multiply attention scores with values
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        
        # Apply final linear layer to get the output
        out = self.fc_out(out)
        return out

class Encoder(nn.Module):
    def __init__(self, model_name):
        super(Encoder, self).__init__()
        self.bert_model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
    def forward(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors = "pt", padding=True)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            
class Matcher(nn.Module):
    def __init__(self, embed_size, heads):
        super(Matcher, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.linear = self.Linear()
        self.embed_size = embed_size
        self.linear1 = nn.Linear(embed_size, 64)
        self.linear2 = nn.linear(64, 2)
    
    def forward(self, input):
        x = SelfAttention(input).mean(dim=1).view(self.embed_size, -1)
        x = nn.functional.relu(self.linear1*(x))
        x = nn.functional.softmax(self.linear2(x))
        
        return x        
        
        
class AICJNet(nn.Module):
    def __init__(self, model_name, embed_size, heads):
        super(AICJNet, self).__init__()
        self.encoder = Encoder(model_name)
        self.matcher = Matcher(embed_size, heads)
        self.embed_size = embed_size
        #self.heads = heads
    
    def forward(self, x):
        x = self.encoder(x)
        out = self.matcher(x)
        return out
    
    
    
        