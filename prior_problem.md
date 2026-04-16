# MatchaTTS training problem:
I made 2 changes to the model:
1. fine resolution (2x mel frames, hop=128) for encoder and MAS; kept hop 256 for Decoder 
2. pre+phoneme+post token scheme (~2x text tokens). 

Now prior_loss reverses direction at epoch 55 and never recovers.
Either feature alone didn't cause this permanent reversal.
I have resumed from epoch 54 or even 49, multiple times, with various LR rates, it happens every time.

I had problems with prior loss in the past, where it jumped up but the recovered, at around epoch 200. 
After the fine resolution change, it started happening much sooner, but I fixed by forcing fp32 in find_alignment().  
I found matmul precision critical — bf16 matmuls caused prior loss explosions in the past.

MAS is not part of the problem. Even after the prior loss shoots up, MAS is almost able to compensate, 
going up and down but generally horizontal, adjusting to the new shot the Encoder sends to it.

The most important questions is why is the optimizer not able to adjust the model weights and get the loss to 
drop from then on. Normally, when a loss grows, the model adjusts after a while. 
The fact that it keeps growing must mean something and could be the key to this problem.

# Background:
The reason I made those 2 changes is that the model struggles with durations.
Matcha must insert a separator in between every 2 phonemes, as the model needs a symbol to assign transitional between phonemes.
Each phoneme p1 is replaced by a tuple of (sep, p1).
Problem is, the exact same separator is inserted everywhere, and the model must guess the duration of the separator 
and the acoustic content only from context. That does not work great.
I realized I can improve this by replacing each phoneme P1 with a tuple of
(pre1, p1, post1). This made the phoneme embeddings table 3 time larger, but allows the model to model transitions much better.
But this is not usable as is, because the hop length of 256 means each symbol has at least 10ms. 
Some consonants are much shorter than 30ms and the model is not able to find alignment.
If some consonants early in the sentence eagerly eat up mel frames, some phonemes later in the sentence will be skipped.
One way to fix it is to run the Encoder and MAS with a hop oof 128, making the duration 5ms and the shortest phoneme 15ms.
So I don't have to run the Decoder (the slowest part) with a hop of 128, I can downsample back just before the Decoder.
This works great, the encoder is able to predict phonemes better, because each transition has its own symbol.
MAS is able to align them better too, since they are shorter and don't eat up frames that don't belong to them.
This is supported by better MCD values (measured at epoch 44) and by subjective listening tests. 
But I get the problem explained above at epoch 54.

# Debugging tools:
- logged values of all tensors involved in MAS computation; all were normal
- compared those values from before the losses shoot up with before they do, nothing looked suspicious.

# Breaktrough:
Trained from epoch 54 without prior loss, and the problem did not manifest. 
Model still calculates MAS, but with the last good prior values from epoch 54. 
This proves the problem is not in MAS, it is in the text_encoder code somewhere, or in the way we calculate the prior loss.

# Solutions attempted:

## I have switched the prior loss formula from L1 to smooth_L1
I used a threshold of 0.07, which is the point where the formula switches from L1 to L2, and trained from scratch. 
I still saw the loss growing at epoch 47. 
But the MCD value at epoch 44 is lower than I ever got, so maybe I should keep the L1_smooth formula.
It is not the solution for the bug described above.

## I fixed the attn_mask in Encoder's forward method
attn_mask = (x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)).bool()
Original did not do .bool(), which allowed padding to participate in attention.
This is a bug, but not the prior loss bug described above.

## I switched the entire training to fp32 temporarily
It did not fix the problem.
I think this proves it is not a numerical instability problem. 

## I have switched the prior loss calculation from the fine to the coarse mel
The problem manifested when I was calculating prior loss by comparing fine resolution ground truth mel to the mel 
assembled from the fine resolution mel frames predicted by the encoder. 
Right after that, we downsample the fine res mel to send it to the Decoder.
I switched to calculating the prior loss by comparing the std res ground truth to the downsampled mel.
The problem did not manifest anymore. 
This shows that the problem is in the very formula of the prior loss:
```python
   prior_loss = torch.sum(torch.abs(y_fine - mu_y_fine) * y_fine_mask)
   prior_loss = prior_loss / (torch.sum(y_fine_mask) * self.n_feats)
```
The loss does not drop smoothly when using the downsampled mel. It zigzags and struggles (but it drops overall).
I don't want to use this computation, but it was a good indication of where the problem occurs.

The 2 changes I made are impacting the prior formulat as follows:
- y_fine, mu_y_fine are larger tensors (1.5x longer phoneme sequences)
shape of y_fine, mu_y_fine is (batch_size, largest_sequence_length, n_feats)
- torch.sum(y_fine_mask) is a 1.5x larger number no matter how small |y_fine - mu_y_fine| is.

Same applies to smooth_l1:
```python
   prior_loss = F.smooth_l1_loss(y_fine * y_fine_mask, mu_y_fine * y_fine_mask, beta=0.07, reduction='sum')
   prior_loss = prior_loss / (torch.sum(y_fine_mask) * self.n_feats)
```

# Relevant training code:
```python
    def forward(self, x, x_lengths, y, y_lengths, y_fine, y_fine_lengths, spks):
        speaker_embedding = self.speaker_embeddings(spks)

        mu_x, logw, x_mask = self.encoder(x, x_lengths, speaker_embedding)
        y_fine_max_length = y_fine.shape[-1]

        y_fine_mask = sequence_mask(y_fine_lengths, y_fine_max_length).unsqueeze(1).to(x_mask)
        attn_mask_fine = x_mask.unsqueeze(-1) * y_fine_mask.unsqueeze(2)

        with torch.autocast(device_type="cuda", enabled=False):
            mu_x = mu_x.float()
            y_fine = y_fine.float() 
            attn_fine = self.find_alignment(attn_mask_fine, mu_x, y_fine)

        mas_durations = torch.sum(attn_fine.unsqueeze(1), -1).squeeze(1)  # (B, T_text)
        
        # I am adding a 2 to make the log-space values greater than 1, because MSE is more forgiving with sub-unitary
        # losses, and more punishing with supra-unitary losses.
        # E.g. 0.6 ** 2 < 0.6, but 1.6 ** 2 > 1.6  
        # This helps Duration Predictor A LOT. We have to compensate for the +2 before synthesis, see inference.py.  
        logw_ = torch.log(2 + mas_durations.unsqueeze(1)) * x_mask

        dur_loss = duration_loss(logw, logw_, x_lengths)

        mu_y_fine = torch.matmul(mu_x, attn_fine.squeeze(1))

        if self.prior_loss:
            prior_loss = F.smooth_l1_loss(y_fine * y_fine_mask, mu_y_fine * y_fine_mask, beta=0.07, reduction='sum')
            prior_loss = prior_loss / (torch.sum(y_fine_mask) * self.n_feats)
        else:
            prior_loss = 0

        mu_y_coarse = downsample(mu_y_fine)

        detached_mu_y_coarse = mu_y_coarse.detach()
        y_max_length = y.shape[-1]
        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        diff_loss = self.decoder.compute_loss(x1=y, mask=y_mask, mu=detached_mu_y_coarse)

        return diff_loss, dur_loss, prior_loss

    def find_alignment(self, attn_mask_fine, mu_x, y_fine):
        with torch.no_grad():
            factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
            y_fine_square = torch.matmul(factor.transpose(1, 2), y_fine ** 2)
            y_fine_mu_double = torch.matmul(-mu_x.transpose(1, 2), y_fine)
            mu_square = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)
            log_prior = y_fine_square - y_fine_mu_double + mu_square + self.mas_const

            attn_fine = maximum_path_gpu(log_prior, attn_mask_fine.squeeze(1).to(torch.int32), log_prior.dtype)

        return attn_fine
```