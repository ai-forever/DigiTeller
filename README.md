# DigiTeller

Русская версия этого документа находится [здесь.](https://github.com/sberbank-ai/DigiTales/blob/main/Readme_Rus.md)

Artificial intelligence technologies are gaining more and more popularity in the creative industries: they are used to write music, create unique art objects, and much more.
We want to invite you to create a solution that will help you create a video story. Algorithms of this kind greatly facilitate or even completely automate the time-consuming process of creating new stories and their visualization.

[![IMAGE ALT TEXT](http://img.youtube.com/vi/CH8hfuGudVA/0.jpg)](http://www.youtube.com/watch?v=CH8hfuGudVA "Digi Teller baseline test")




We provide a baseline consisting of google collab notebooks:


[Part 1](https://colab.research.google.com/drive/1mk7U9jG8w8TmByEB9VKjt5Z7h_DVIOz2?usp=sharing)

 * dataset for training [here](https://drive.google.com/file/d/1iVBMLz_9MbFEp1wL2QjmVNkbnFs6iBxR/view?usp=sharing)

[Part1 A](https://colab.research.google.com/drive/1mk7U9jG8w8TmByEB9VKjt5Z7h_DVIOz2?usp=sharing)

[Part1 Songs](https://colab.research.google.com/drive/1Imu55S5In67ouALiqwvvyZthjUYVcJ2t?usp=sharing)

datset for training 
  * [lyrics](https://drive.google.com/file/d/1y2MexkTjwFbhePj3HUslY1gmxOmpYzy3/view?usp=sharing) 

  * [chords](https://drive.google.com/file/d/1Ak2H78XW31oNEtM3m2jL1UJZ5fFpU3tt/view?usp=sharing)

Trained on giant Ficbook dataset, description and data availible 

[Part 2](https://colab.research.google.com/drive/1BzrSLZfXsMrxc1c3A9tp-_73qo-z2qjU?usp=sharing)

[Part 2 A](https://colab.research.google.com/drive/1PdZ2Ki-HJh9HONVykTOBcjmOSCe4MhwY?usp=sharing)

Silero V2 features, only Ksenya

[Part 3](https://colab.research.google.com/drive/16XBLgWJ-aYQMmu6OADSoO8a8rCf_ZQL5?usp=sharing)

[Part 3a](https://colab.research.google.com/drive/1ahU-a4bicqv3eUv_cKBax9pIeJiwHjFl?usp=sharing)

[Part 3B](https://colab.research.google.com/drive/10XqQM1rmwmEbrDBTDRT7SN8vX_OZDSmE#scrollTo=ukxFcvUk_epI)

[Pixel draw](https://colab.research.google.com/github/tg-bomze/collection-of-notebooks/blob/master/Text2PixelArt.ipynb)

[MT CLIP + VQGAN](https://colab.research.google.com/drive/1MpVMCq2G_LIeqwOvWy-ECg-j_M3VLCbv?usp=sharing)

[Half Pression Diffusuion Model](https://colab.research.google.com/drive/18YfYqAMvUul6cn0I949F8d0gTcZW-L4q?usp=sharing)
Possible to run on T4 and K80 gpu

Implemented new features such as Part od Speech filtering

[Part 4](https://colab.research.google.com/drive/1ARNsECcfkWoJoxl25_G9xp5vOfS2HgXI?usp=sharing)

The task consists of 5 large blocks, more details about them are described below

## Block 1. Text Generation
Huge language models (like GPT-3) surprise us more and more with their capabilities. While business confidence in them is not yet sufficient to present them to their customers, these models demonstrate the beginnings of intelligence that will accelerate the development of automation and the capabilities of "smart" computing systems. Let's take the aura of mystery out of GPT-3 and find out how it learns and how it works.
The trained language model generates text. We can also send some text to the input of the model and see how the output changes. The latter is generated from what the model has "learned" during the training period by analyzing large amounts of text.
Learning is the process of transferring a large amount of text to a model. For GPT-3, this process is complete and all the experiments you can see are running on the already trained model. It was estimated that the training would take 355 GPU-years (355 years of training on a single graphics card) and cost $ 4.6 million.

![1-1](https://user-images.githubusercontent.com/57997673/119852987-75b74f00-bf18-11eb-9835-2a3708f94252.png)
At the input of the model, we give one example (display only features) and ask her to predict the next word of the sentence.
The model's predictions will be wrong at first. We calculate the prediction error and update the model until the predictions improve.
And so several million times.

![1-2](https://user-images.githubusercontent.com/57997673/119853198-a6978400-bf18-11eb-8b57-a60aef2f360f.png)
Important GPT-3 computations take place inside a stack of 96 layers of the Transformer decoder.
See all these layers? This is the very “depth” of “deep learning”.
Each layer has its own 1.8 billion parameters to compute. This is where all the "magic" happens. At the top level, this process can be depicted as follows:

![1-3](https://user-images.githubusercontent.com/57997673/119853333-c75fd980-bf18-11eb-9b76-d8bc694b9d4b.png)

## Block 2. Text2Speech model
The usage of concatenative TTS is limited due to high data requirements and development time. Therefore, a statistical method has been developed that investigates the very nature of the data. It generates speech by combining parameters such as frequency, amplitude spectrum, etc.
Parametric synthesis consists of the following steps.
First, linguistic features are extracted from the text, for example, phonemes, duration, etc.
Then, for a vocoder (a system that generates waveforms), features are extracted that represent the corresponding speech signal: cepstrum, frequency, linear spectrogram, chalk spectrogram.
These hand-tuned parameters, along with linguistic features, are fed into the vocoder model, which performs many complex transformations to generate the sound wave. In this case, the vocoder evaluates speech parameters such as phase, prosody, intonation, and others.
If we can approximate the parameters that define speech at each of its units, then we can create a parametric model. Parametric synthesis requires significantly less data and hard work than concatenative systems.

![2-1](https://user-images.githubusercontent.com/57997673/119853730-1f96db80-bf19-11eb-9cb7-2796d6a41c54.png)

## Block 3. Music Generation
Music Transformer is an attention-based neural network from OpenAI that can generate music with improved long-term coherence.
The model uses an event-based view that allows us to directly generate expressive performances. There are 388 events in the model: 128 events of turning on notes of different heights, 128 events of turning off notes of different heights, 100 events of transition to the next time interval and 32 events of changing the speed.
The model uses a relative attention mechanism that explicitly modulates attention depending on how far apart the two tokens are, the model can focus more on relational functions. Relative self-attention also allows the model to generalize beyond the length of the training examples, which is not possible with the original Transformer model.

![2-2](https://user-images.githubusercontent.com/57997673/119854082-70a6cf80-bf19-11eb-8a2a-1aae4cc67be0.png)


## Bonus. Jukebox neural network music generation

Jukebox is a neural network from OpenAI that generates music, including elementary singing, as raw sound in a variety of genres and performer styles.
The Jukebox autoencoder model compresses audio into discrete space using a quantization-based approach called VQ-VAE. The Jukebox uses an upgraded VQ-VAE.
Three levels are used in the VQ-VAE shown below, which compress 44 kHz unprocessed audio at 8x, 32x and 128x, respectively. Downsampling loses most of the audio detail and sounds noticeably noisy as you go further down the levels. However, it retains important information about pitch, timbre, and volume.

### Compress

![2-3](https://user-images.githubusercontent.com/57997673/119854417-b82d5b80-bf19-11eb-9768-fec125905aac.png)

After all the a priori are trained, the music is generated at the top level and then upsampled with upsamplers and decoded back into the raw audio space using a VQ-VAE decoder to sample new songs.

### Generate

![2-4](https://user-images.githubusercontent.com/57997673/119854500-c9766800-bf19-11eb-9969-8c88a5073b0c.png)

To pay attention to the lyrics, an encoder is added to create a representation of the lyrics, and attention levels are added that use requests from the music decoder to handle keys and values from the lyrics encoder. After training, the model learns to align the text more accurately.

## Block 4. Clip generation, CLIP + BIGGAN

CLIP (Contrastive Language - Image Pre-training) is based on a large body of work on zero-shot transfer, natural language control and multimodal learning. The idea of learning with zero-data originated in the 80s, but until recently was mainly studied in the field of computer vision as a way to generalize to the categories of invisible objects.
It was critical to use natural language as a flexible predictive space to generalize and communicate. In 2013, Richer Socher and coauthors at Stanford developed a proof of concept by training a CIFAR-10 model to make predictions in the embedding space of word vectors, and showed that the model can predict two invisible classes. In the same year, DeVISE12 scaled up this approach and demonstrated that it was possible to fine-tune the ImageNet model so that it could be generalized to correctly predict features beyond the original 1000 training set.

We show that scaling up a simple pre-training problem is sufficient to achieve competitive zero-shot performance on a large number of image classification datasets. Our method uses a widely available source of control: text combined with images found on the Internet. This data is used to create the following CLIP proxy training problem: from a given image, predict which of 32,768 randomly selected text fragments were actually associated with it in our dataset.
Our intuition dictates that in order to accomplish this task, CLIP models must learn to recognize a wide range of visual concepts in images and associate them with their names. As a result, CLIP models can be applied to nearly arbitrary visual classification problems. For example, if the objective of the dataset is to classify photographs of dogs and cats, we check for each image whether the CLIP model predicts the textual description "dog photograph" or "cat photograph" is more likely to be paired. with this.

![3-1](https://user-images.githubusercontent.com/29739660/119980287-c1273700-bfc4-11eb-98cd-1b1189f4af86.png)
)

CLIP pre-trains an image encoder and a text encoder to predict which images were associated with which texts in our dataset. We then use this behavior to turn CLIP into a zero-shot classifier. We convert all dataset classes to captions, such as "dog photo", and predict the caption class. CLIP rates the best pairs with a given image.

## Block 4A. Clip generation, CLIP +Taming transformer

![image](https://user-images.githubusercontent.com/29739660/119980344-d308da00-bfc4-11eb-9878-8d0ccb390a44.png)
Designed to learn long-range interactions on sequential data, transformers continue to show state-of-the-art results on a wide variety of tasks. In contrast to CNNs, they contain no inductive bias that prioritizes local interactions. This makes them expressive, but also computationally infeasible for long sequences, such as high-resolution images. We demonstrate how combining the effectiveness of the inductive bias of CNNs with the expressivity of transformers enables them to model and thereby synthesize high-resolution images. We show how to (i) use CNNs to learn a context-rich vocabulary of image constituents, and in turn (ii) utilize transformers to efficiently model their composition within high-resolution images. Our approach is readily applied to conditional synthesis tasks, where both non-spatial information, such as object classes, and spatial information, such as segmentations, can control the generated image. In particular, we present the first results on semantically-guided synthesis of megapixel images with transformers.

## Block 5. Increaising image resolution, SuperResolution

To solve the single image super resolution problem, GANs were used, namely ESRGANs. ESRGAN (Enchanced-SRGAN) is an improved version of the SRGAN neural network, so let's talk about it first.

SRGAN is a generative adversarial neural network for the task of obtaining high quality photorealistic images from poor quality images.
Generative adversarial networks are an algorithm built on a combination of two neural networks, one of which (generator) generates patterns, and the other (discriminator) tries to distinguish correct ("genuine") patterns from incorrect ones.
The generator and discriminator architecture used in SRGAN:

<img width="863" alt="srgan_architecture" src="https://user-images.githubusercontent.com/57997673/120758283-4e651100-c51a-11eb-8b5d-4c2a37135b3f.png">

SRGAN's success in the super resolution task is due to the following features:
Generative adversarial networks allow for more realistic images than neural networks based on MSE optimizations between pixels. The pixel-optimized MSE models averaged the textures, making them overly smooth. The use of GANs, thanks to a discriminator that learns to distinguish fake generated images from real ones, allows you to generate an image from a variety of photorealistic options without averaging all textures.

![natural_manifold](https://user-images.githubusercontent.com/57997673/120758400-7785a180-c51a-11eb-82c2-1192deaf3608.png)

The second important feature was the use of Perceptual loss. In the process of training the neural network, the optimization for MSE between pixels was replaced by perceptual loss, which is an MSE calculated in the feature space of a deep convolutional network (in particular, VGG19). This error is more invariant to changes in the pixels of the image, which gives the generator more freedom to change the image.

### What has been improved in ESRGAN
ESRGAN differs from its predecessor in two main innovations:

-Changed the generator architecture, replacing residual blocks with larger Residual-in-Residual Dense blocks (RRDB) without batch normalization.
-The usual discriminator has been replaced with a relativistic discriminator, which predicts the likelihood that the original image is more realistic than the generated one.


Solution based on
* https://github.com/lucidrains/big-sleep
* https://github.com/snakers4/silero-models
* https://openai.com/blog/clip/
* https://openai.com/blog/jukebox/
* https://magenta.tensorflow.org/music-transformer
* https://arxiv.org/pdf/1809.00219.pdf
