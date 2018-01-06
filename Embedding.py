import torch.nn as nn
import torch

class Embedding(nn.Embedding):
    def reset_parameters(self):
        print("Use uniform to initialize the embedding")
        self.weight.data.uniform_(-0.01, 0.01)

        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

class ConstEmbedding(nn.Module):
    def __init__(self, pretrained_embedding, padding_idx=0):
        super(ConstEmbedding, self).__init__()
        self.vocab_size = pretrained_embedding.size(0)
        self.embedding_size = pretrained_embedding.size(1)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx, sparse=True)
        self.embedding.weight = nn.Parameter(pretrained_embedding, requires_grad=False)

    def cuda(self, device_id=None):
        """
           The weights should be always on cpu
       """
        return self._apply(lambda t: t.cpu())

    def forward(self, input):
        """
           return cpu tensor
       """
        # is_cuda = next(input).is_cuda
        is_cuda = input.is_cuda
        if is_cuda: input = input.cpu()
        self.embedding._apply(lambda t: t.cpu())

        x = self.embedding(input)
        if is_cuda: x = x.cuda()

        return x

class VarEmbeddingCuda(nn.Module):
    def __init__(self, pretrained_embedding, padding_idx=0):
        super(VarEmbeddingCuda, self).__init__()
        self.vocab_size = pretrained_embedding.size(0)
        self.embedding_size = pretrained_embedding.size(1)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.embedding.weight = nn.Parameter(pretrained_embedding, requires_grad=True)
        # self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        # self.embedding.weight.requires_grad = True

    def forward(self, input):
        x = self.embedding(input)
        return x

class VarEmbeddingCPU(nn.Module):
    def __init__(self, pretrained_embedding, padding_idx=0):
        super(VarEmbeddingCPU, self).__init__()
        self.vocab_size = pretrained_embedding.size(0)
        self.embedding_size = pretrained_embedding.size(1)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.embedding.weight = nn.Parameter(pretrained_embedding, requires_grad=True)
        # self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        # self.embedding.weight.requires_grad = True

    def forward(self, input):
        is_cuda = input.is_cuda
        if is_cuda: input = input.cpu()
        self.embedding._apply(lambda t: t.cpu())

        x = self.embedding(input)
        if is_cuda: x = x.cuda()
        return x


class LSTM(nn.LSTM):
    def reset_parameters(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                for i in range(4):
                    nn.init.orthogonal(self.__getattr__(name)[self.hidden_size*i:self.hidden_size*(i+1),:])
            if "bias" in name:
                nn.init.constant(self.__getattr__(name), 0)


def load_pretrained_emb_zeros(path, text_field_words_dict, set_padding=False):
    padID = text_field_words_dict['<pad>']
    embedding_dim = -1
    with open(path, encoding='utf-8') as f:
        for line in f:
            line_split = line.strip().split(' ')
            if len(line_split) == 1:
                embedding_dim = line_split[0]
                break
            elif len(line_split) == 2:
                embedding_dim = line_split[1]
                break
            else:
                embedding_dim = len(line_split) - 1
                break
    word_count = len(text_field_words_dict)
    print('The number of words is ' + str(word_count))
    print('The dim of pretrained embedding is ' + str(embedding_dim) + '\n')
    embeddings = np.zeros((word_count, embedding_dim))
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            values = line.split(' ')
            index = text_field_words_dict.get(values[0])   # digit or None

            if index:
                vector = np.array(values[1:], dtype='float32')
                embeddings[index] = vector

    return torch.from_numpy(embeddings).float()

def load_pretrained_emb_avg(path, text_field_words_dict, set_padding=False):
    padID = text_field_words_dict['<pad>']
    embedding_dim = -1
    with open(path, encoding='utf-8') as f:
        for line in f:
            line_split = line.strip().split(' ')
            if len(line_split) == 1:
                embedding_dim = line_split[0]
                break
            elif len(line_split) == 2:
                embedding_dim = line_split[1]
                break
            else:
                embedding_dim = len(line_split) - 1
                break
    word_count = len(text_field_words_dict)
    print('The number of words is ' + str(word_count))
    print('The dim of pretrained embedding is ' + str(embedding_dim) + '\n')
    embeddings = np.zeros((word_count, embedding_dim))

    inword_list = []
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            values = line.split(' ')
            index = text_field_words_dict.get(values[0])   # digit or None
            if index:
                vector = np.array(values[1:], dtype='float32')
                embeddings[index] = vector
                inword_list.append(index)

    sum_col = np.sum(embeddings, axis=0)/len(inword_list)   # 按列求和，再求平均
    for i in range(len(text_field_words_dict)):
        if i not in inword_list and i != padID:
            embeddings[i] = sum_col

    return torch.from_numpy(embeddings).float()


