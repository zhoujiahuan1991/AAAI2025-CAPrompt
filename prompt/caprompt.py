import torch
import torch.nn as nn


class EPrompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',
                 num_layers=1, use_prefix_tune_for_e_prompt=False, num_heads=-1, same_key_value=False,):
        super().__init__()

        self.length = length
        #self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.num_layers = num_layers
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt
        self.num_heads = num_heads
        self.same_key_value = same_key_value

        if self.use_prefix_tune_for_e_prompt:
            assert embed_dim % self.num_heads == 0
            if self.same_key_value:
                prompt_pool_shape = (self.num_layers, 1, self.pool_size, self.length, 
                                    self.num_heads, embed_dim // self.num_heads)

                if prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)
                self.prompt = self.prompt.repeat(1, 2, 1, 1, 1, 1)
            else:
                prompt_pool_shape = (self.num_layers, 2, self.pool_size, self.length, 
                                    self.num_heads, embed_dim // self.num_heads)
                if prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape)) # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
                    nn.init.uniform_(self.prompt, -1, 1)
        else:
            prompt_pool_shape = (self.num_layers, self.pool_size, self.length, embed_dim)  # TODO fix self.num_layers = 1
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)
                
       
            
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, prompt_weight=None, task_id = 0):

        out = dict()

        if self.use_prefix_tune_for_e_prompt:

            #index = idx[0][0]
            index = task_id
            #print(index)
            batched_prompt_raw = torch.einsum("bp,ndplhe->ndblhe", prompt_weight[:,:index], self.prompt[:,:,:index].detach().clone()) +torch.einsum("bp,ndplhe->ndblhe", prompt_weight[:,index:], self.prompt[:,:,index:]) # num_layers, 2, B, top_k, length, C
            batched_prompt_raw = batched_prompt_raw.unsqueeze(3)
            num_layers, dual, batch_size, top_k, length, num_heads, heads_embed_dim = batched_prompt_raw.shape
            # print(top_k)
            batched_prompt = batched_prompt_raw.reshape(
                num_layers, batch_size, dual, top_k * length, num_heads, heads_embed_dim
            )
            
                
        out['batched_prompt'] = batched_prompt

        return out
