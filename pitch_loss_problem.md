The original code was calculating the following losses:
1. diffusion loss
2. duration loss
3. prior loss

I added a pitch extractor and I am running training with all 4 losses aggregated.
The trainer computes the losses separately on the training dataset and on the validation dataset.
I am now at step 60000.
All training and validation losses are trending downward, with one exception.

The *pitch loss* on the *validation dataset* started to trend upward after step 20000. 
It is still trending downward on the training dataset.

This is a classic case of overfitting during training.
The pitch extractor is memorizing pitch patterns in the training data rather than learning generalizable pitch features. 
After step 20000, it started fitting to training-specific characteristics (like speaker idiosyncrasies, recording conditions, or dataset biases) 
that don't transfer to the validation set.

It happens only on the pitch loss, probably because the other components are:
- More constrained by the model architecture
- Learning more robust features
- Have better regularization implicitly
- Or are less prone to overfitting by nature

What I can do:
- Reduce pitch loss weight - It may be dominating the gradient updates
- Add regularization - Dropout in the pitch predictor, weight decay
- Data augmentation - Pitch shifting, formant preservation techniques
- Simplify the pitch extractor - It might be too expressive/complex
- Check pitch targets - Are they noisy? Consider smoothing or using continuous wavelet transform features
- Early stopping - Consider using the model checkpoint from around step 20000 for pitch prediction

The fact that your other losses are healthy suggests the overall model is fine.