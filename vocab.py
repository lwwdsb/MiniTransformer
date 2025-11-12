PAD = 0
BOS = 1
EOS = 2

def build_vocab(sentences):
    #Collect unique words
    words = set() # to hold unique words
    for s in sentences:
        for w in s.split():
            words.add(w)
    
    #Initialize vocab with special tokens
    word2idx = {"<pad>": PAD, "<bos>": BOS, "<eos>": EOS}
    idx2word = {PAD: "<pad>", BOS: "<bos>", EOS: "<eos>"}
    
    for i, w in enumerate(sorted(list(words)), start = 3):
        word2idx[w] = i
        idx2word[i] = w
        
    return word2idx, idx2word
  
def encode(sentences, word2idx, max_len):
    tokens = [word2idx[w] for w in sentences.split()]
    tokens = [BOS] + tokens + [EOS]
    if len(tokens) < max_len:
        tokens = tokens + [PAD] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return tokens
      