# Problem 
MatchaTTS neds separators inserted in between all phonemes.
The reason as I understand it is for the model to assign transitional sounds to the separators. 
Like for example when pronouncing you, the sound of "y" morphs into the sound of "o", and later the sound of "o" 
will morph into the sound of "u".

Input text:      <Drop the two when you add the figures.>
Phonetised text: < |d|ɹ|ˈ|ɑ|ː|p| |ð|ə| |t|ˈ|u|ː| |w|ɛ|n| |j|u|ː| |ˈ|æ|d| |ð|ə| |f|ˈ|ɪ|ɡ|j|ɚ|z|.>

What I don't like about that "blind insertion", is that separators got between annotations and their annotated phoneme, 
which does not make sense, there is no justification for it. 
The model allocates a sound and at least one frame (uses MAS) to each symbol, including to annotations.

I tried many alternatives.

1. no separators at all:
   Phonetised text: < dɹˈɑːp ðə tˈuː wɛn juː ˈæd ðə fˈɪɡjɚz.>
   Model does not learn speech at all.

2. group annotations with their annotated phoneme, and insert separators only between groups:
   Phonetised text: < |d|ɹ|ˈɑː|p| |ð|ə| |t|ˈuː| |w|ɛ|n| |j|uː| |ˈæ|d| |ð|ə| |f|ˈɪ|ɡ|j|ɚ|z|.>
   Sounds better than with blind inserts, but makes pronunciation mistakes.

3. no separators, but replace every voiced phoneme V with a tuple (preV, V, postV):
   < |d||ɹ|ˈ|ɑ|ː|p| |ð||ə| |t|ˈ|u|ː |w||ɛ||n| |j||u|ː ˈ|æ||d| |ð||ə| |f|ˈ|ɪ||ɡ||j||ɚ||z|.>
   I cannot show you the actual symbols, but I had 3 times as many symbols in this version, the vowels and consonants,

4. a pre and a post for each vowel and consonant.
   Sounds even better, but again makes some pronunciation mistakes.
   Especially around short consonants.

The reason it sounds better when choosing option 2, I think, is that we don't get a lot of forced frames between vowels. 
Any annotation costs 3 frames: "|ˈ|"  or "|ː|" and a frame is 11ms, which is too long.

The model is not inherently broken, just too blunt.

# Possible solutions

1. Shorter frame length 
Lower hop length from 256 to 192 will shorten frame length from 10.6ms to 8ms.
Model will get slower so maybe a better approach would be to use 16KHz audio with a hop of 128.

2. Double Resolution for MAS
I could run Encoder, MAS and Duration Predictor with a hop length of 128.
MAS will then align segments with a hop of 128 against the high resolution ground truth mel.
MAS will be able to find positions that are multiples of 5.33ms.
Duration Predictor will learn to generate those fine-grained positions.
I will assemble the fine resolution mel by repeating each predicted mel frame a number of time as computed by MAS at the fine resolution. 
Prior Loss will be calculated on the fine resolution mel.
To send it to the Decoder, though, I will have to downsample it by a factor of 2, to get standard resolution mel. 
This way the Encoder will still work with a hop length of 256.
The Vocoder too, at inference time will still run on mels with a hop of 256. 

# Results

I implemented Solution 2. Speech sounds a bit sharper indeed.

It does not fix all the problems. I still have 3 problems:
Problem 1: skipped consonants (rare)
Problem 2: mispronounced vowels (rare) or consonants (very rare)
Problem 3: mispronunciation on short utterances (very frequent)

I think the weak link in this model is MAS and the duration predictor.

The predictor sometimes allocates too few frames to consonants.  
I am clamping to 2 high res frames at inference, but it does not fix Problem 1 entirely. 
If I clamp to 1 frame, speech sounds even sharper, but problem eis more frequent. 
If I clamp to 3 high res frames, it elongates some words unnaturally. 

Also, Problem 2 is probably because of wrong alignments. Not sure.

I don't know what causes Problem 3. 
