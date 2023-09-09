import torch
import torch.nn as nn
import transformers
from transformers import BertTokenizer, BertModel
import torch.nn.functional as f

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
    
    def forward(self, values, keys, query, masks):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        values = values.view(N, value_len, self.heads, self.head_dim)
        keys = values.view(N, key_len, self.heads, self.head_dim)
        query = query.view(N, query_len, self.heads, self.head_dim)
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)
        
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        attention = torch.nn.functional.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        # Multiply attention scores with values
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        
        # Apply final linear layer to get the output
        out = self.fc_out(out)
        return out

class Encoder(nn.Module):
    def __init__(self, model_name, device):
        super(Encoder, self).__init__()
        self.bert_model = BertModel.from_pretrained(model_name).to(device)
        
    def forward(self, inputs):
        size, attr_num, max_len = inputs.squeeze().shape
        inputs = inputs.view(size*attr_num, max_len)
        with torch.no_grad():
            outputs = self.bert_model(input_ids=inputs)
        return outputs.last_hidden_state
            
class Matcher(nn.Module):
    def __init__(self, embed_size, heads, hidden_size):
        super(Matcher, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.embed_size = embed_size
        self.linear1 = nn.Linear(embed_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 2)
    
    def forward(self, inputs, masks, attr_num):
        size, max_len, embed_size = inputs.shape
        inputs = inputs.view(-1, 2*attr_num*max_len, embed_size)
        x = self.attention(inputs, inputs, inputs, masks).mean(dim=1).view(self.embed_size, -1).transpose(0,1)
        last = nn.functional.relu(self.linear1(x))
        out = nn.functional.softmax(self.linear2(last))
        
        return out, last
        
        
class AICJNet(nn.Module):
    def __init__(self, model_name, embed_size, heads, device, hidden_size):
        super(AICJNet, self).__init__()
        self.encoder = Encoder(model_name, device)
        self.matcher = Matcher(embed_size, heads, hidden_size)
        self.embed_size = embed_size
    
    def forward(self, inputs, masks):
        size, attr_num, max_len = inputs.squeeze().shape
        x = self.encoder(inputs)
        out, last = self.matcher(x, masks, attr_num)
        return out, last
    
    
    
        