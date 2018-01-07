import torch.nn as nn
import torch

class Embedding(nn.Embedding):
    def reset_parameters(self):
        print("Use uniform to initialize the embedding")
        # self.weight.data.normal_(0, 1)
        # if self.padding_idx is not None:
        #     self.weight.data[self.padding_idx].fill_(0)

        self.weight.data.uniform_(-0.01, 0.01)
        # print(self.padding_idx)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

class ConstEmbedding(nn.Module):
    def __init__(self, pretrained_embedding, padding_idx=0, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False):
        super(ConstEmbedding, self).__init__()
        self.vocab_size = pretrained_embedding.size(0)
        self.embedding_size = pretrained_embedding.size(1)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx, max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse)
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
    def __init__(self, pretrained_embedding, padding_idx=0, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False):
        super(VarEmbeddingCuda, self).__init__()
        self.vocab_size = pretrained_embedding.size(0)
        self.embedding_size = pretrained_embedding.size(1)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx, max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse)
        self.embedding.weight = nn.Parameter(pretrained_embedding, requires_grad=True)
        # self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        # self.embedding.weight.requires_grad = True

    def forward(self, input):
        x = self.embedding(input)
        return x

class VarEmbeddingCPU(nn.Module):
    def __init__(self, pretrained_embedding, padding_idx=0, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False):
        super(VarEmbeddingCPU, self).__init__()
        self.vocab_size = pretrained_embedding.size(0)
        self.embedding_size = pretrained_embedding.size(1)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx, max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse)
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




