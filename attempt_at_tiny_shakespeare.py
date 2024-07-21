import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters

batch_size = 37
block_length = 8
max_iters = 3000
eval_interval = 300
learning_rate = le-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

#---------------------

torch.manual_seed(1337)


file_path = 'E:\\python\\Tiny Shakesphere Project\\tiny-shakespeare.txt'

with open(file_path, 'r') as file:
  # Read the entire content as a string
  text_data = file.read()


# I am trying to find out a list of all the unique characters that construct the tiny_shakespeare dataset

chars = sorted(list(set(text_data)))
vocab_size = len(chars)
# here we are trying to 'tokenize' the text, in other words create a one to one map of characters to integers
stoi = {ch:i for i,ch in enumerate(chars)} #remember the chars var above that contains list of all characters? we create look up table for set of characters in tiny_shakespeare data
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder takes a string and gives list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # this is doing extactly reverse of the above step, decoder takes list of integers and outputs a string

# we map each character individually to endocde and to decode we carry out the same process in reverse order
# this is merely one technique to create mapping, there could be many other possible ways to carry out this step, for example: google uses 'sentecepiece' a sub-word unit tokenizer, OpenAI uses tiktoken

#----------
'''print(encode("hi there"))
#print(decode(encode("hi there")))
#print(''.join(chars))
print(vocab_size)

#data = torch.tensor(encode(text_data), dtype=torch.long) #we have taken all of the characters in dataset and wrap it around tensor using pytorch library
#print(data.shape, data.dtype)
#print(data[:1000]) #the 1000 characters we saw ealier, for GPT they will look like the following, it is identical translation in form of integers of the same 1000 characters'''

#----------


#Now we will split our whole dataset into training dataset and validation dataset. It is self explanatory, the training dataset will be used to train our language model and the validation dataset will be used to test it to the novel content.
data = torch.tensor(encode(text_data), dtype=torch.long)
n= int(0.8*len(data)) #80% of the dataset will be training dataset and rest of it will be validation
train_data = data[:n]
val_data = data[n:] # this is done to solve the problem of 'overfitting', we don't want the language model to be exactly like the dataset, rather we seek a very close similarity and proximity to how dataset actually is so that when the content is produced, it will be not exact copy of dataset.

#now we will start feeding the text to transformer, but we do not do this all at once because it is computationally inefficient, we do it in chunks or batches.


#train_data[:block_length + 1] 
#here, 18 is followed by 47, when both 18,47 are present they are to be followed by 56, when all three are present 57 follows them etc.
#----------
'''x = train_data[:block_length]
y = train_data[1:block_length+1]
for t in range(block_length):
  context = x[:t+1]
  target = y[t]
  print(f"when input is {context} the target is: {target}")
'''
#-----------
def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_length, (batch_size,))
  x = torch.stack([data[i:i+block_length] for i in ix])
  y= torch.stack([data[i+1:i+block_length+1]for i in ix])
  return x , y


@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval{}:
  for split in ['train','val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = model(X,Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

'''xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)'''


'''for b in range (batch_size):
  for t in range(block_length):
    context = xb[b, :t+1]
    target = yb[b,t]
    print(f"when input is {context.tolist()} the target: {target}")

print(xb) #this is the output to the transformer'''




#we will start to implement the bigram language model

class BigramLM(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
  #each token  here will read the logits required for the next token in line from lookup table


  def forward(self, idx, targets=None):

    logits = self.token_embedding_table(idx)
    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    # Remove the duplicate loss calculation
    # loss = F.cross_entropy(logits, targets)
    return logits, loss

  # De-indent the generate function to make it a method of the class
  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      # Call forward with targets=None since we're generating
      logits, _ = self(idx, targets=None)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      # Remove duplicate line
      # probs = F.softmax(logits, dim=-1)
      # idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx

model = BigramLM(vocab_size)
m = model.to(device)


#creating pyTorch optimizer
optimizer = torch.optm.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

  if iter % eval_interval == 0:
    losses = estimate_loss()
    print(f"step {iter} : train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

  xb, yb = get_batch('train')
  logits, loss = m(xb,yb) # This call now correctly receives both logits and loss
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()


#generating from context
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))




#optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)


'''for steps in range(11000):

  xb,xy = get_batch('train')

  logits, loss = m(xb, xy)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

  print(loss.item())

print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))'''