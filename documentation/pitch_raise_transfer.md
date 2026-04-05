Try to transfer the pitch raise feature to Bella and Nicole using simple Speaker Embedding Arithmetics.
You can try "embedding surgery" during inference too.
1. Calculate the average embedding of your "Question" speakers ($E_{q}$).
2. Calculate the average embedding of your "Flat" speakers ($E_{f}$).
3. Calculate the difference: $\Delta = E_{q} - E_{f}$.
4. At inference, try feeding the model a modified embedding for your flat speakers:   
   $E_{new} = E_{flat\_original} + (\alpha \cdot \Delta)$   
   where $\alpha$ is a small scaling factor.
5. This nudges the speaker's identity toward the space where "question-asking" behavior is common.