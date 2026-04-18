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

# The fix:
I just got to epoch 139 with no problems yet. Here are the changes I made since the last attempt:

- Lowered weight decay from 1e-2 to 1e-3 
I trained with 1e-2 successfully in the past. Now the param norm grows rapidly.
Not sure if this helps or not. I thought that maybe the decay is erasing the model.

- I started with a smooth_L1 with a Beta of 0.3 and I was logging prior residual quantiles. 
At epoch 49 I stopped, checked the p50 residual and found it was between 0.4 and 0.5.
I changed Beta to 0.04 and resumed training.

- Removed n_feats from the prior normalization formula denominator.
This makes the loss 100 times larger than previous attempt.

Here's my guess of what could have been the solution:
1. The new tokenization scheme made, the number of symbols per sequence 1.5x larger.
The sum of y_fine_mask in the denominator proportional to sequence length. When it grows, the loss gets *smaller*.
The loss is calculated by summing up the error between predicted and ground truth mels on *each mel bine* of *each mel frame* of each sample in the batch. 
At first, the error is large, resulting in a strong signal to drive the loss down. 
By epoch 50, *the error gets smaller and smaller*, as the model learns, but the *denominator stays large*.
Because I removed the denominator, the loss is now larger (around 5).

2. The switch to Smooth L1 was also good, because a pure L1 of that magnitude would have been too blunt.
L1 keeps the same gradients to the very end, it punishes small errors too much. Smooth L1 still punishes big errors
but it returns small gradients for small errors. 

The L1 loss is too blunt, model continues to punish even the estimations that are very close to the truth.
At some point, the gradients will make the model jump to a state that makes some mistakes, and won't be able 
to correct them, because of that. If the MAS alignment will change, the model will respond with more corrections in 
the wrong direction. Any jump in prior that is big enough to cause alignment to change will push the model into a 
positive feedback loop it cannot escape.
And this is not new, it was probably always a problem with this model, but now I just made it easier to reproduce

## Update:
I still saw a jump in the prior loss at epoch 94, but the model recovered and loss continued to drop.
After that, the loss was a bit more wiggly than in the first 94 epochs, similar looking to the curve when I was using an L1 formula. 
I think a Beta of 0.5 might smooth out the loss curve more, if needed.