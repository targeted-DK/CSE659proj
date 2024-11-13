from info_nce import InfoNCE, info_nce
import torch

loss = InfoNCE()
batch_size, embedding_size = 32, 128
query = torch.randn(batch_size, embedding_size)

print("Tensor : ", query)


print("Tensor : ", query.size)
positive_key = torch.randn(batch_size, embedding_size)
output = loss(query, positive_key)

loss = InfoNCE(negative_mode='unpaired') # negative_mode='unpaired' is the default value
batch_size, num_negative, embedding_size = 32, 48, 128
query = torch.randn(batch_size, embedding_size)
positive_key = torch.randn(batch_size, embedding_size)
negative_keys = torch.randn(num_negative, embedding_size)
output = loss(query, positive_key, negative_keys)

loss = InfoNCE(negative_mode='paired')
batch_size, num_negative, embedding_size = 32, 6, 128
query = torch.randn(batch_size, embedding_size)
positive_key = torch.randn(batch_size, embedding_size)
negative_keys = torch.randn(batch_size, num_negative, embedding_size)
output = loss(query, positive_key, negative_keys)