This is a pytorch implementation of FFTNet described [here](http://gfx.cs.princeton.edu/pubs/Jin_2018_FAR/).
Work in progress.

## Quick Start

1. Install requirements
```
pip install -r requirements.txt
```

2. Download [CMU_ARCTIC](http://festvox.org/cmu_arctic/) dataset.

3. Train the model and save. The default parameters are pretty much the same as int the original paper. 
Raise the flag _--preprocess_ when execute the first time.

```
python train.py \
    --preprocess \
    --wav_dir your_downloaded_wav_dir \
    --data_dir preprocessed_feature_dir \
    --model_file saved_model_name \
```

4. Use trained model to decode/reconstruct a wav file from the mcc feature.

```
python decode.py \
    --infile wav_file
    --outfile reconstruct_file_name
    --data_dir preprocessed_feature_dir \
    --model_file saved_model_name \
```

[FFTNet_generator](FFTNet_generator.py) and [FFTNet_vocoder](FFTNet_vocoder.py) are two files I used to test the model 
workability using torchaudio yesno dataset.

## Current result

There are some files decoded in the [samples](samples) folder. 

## Differences from paper

* window size: 400 >> depend on minimum_f0 (cuz I use pyworld to get f0 and mcc coefficients)

## TODO

- [x] Zero padding.
- [x] Injected noise.
- [x] Voiced/unvoiced conditional sampling.
- [x] Post-synthesis denoising.

## Notes

* I combine two 1x1 convolution kernel to one 1x2 dilated kernel.
This can remove redundant bias parameters and accelerate total speed.
* The author said in the middle layers the channels size are 128 not 256.
* My model will get stuck at the begining (loss aroung 4.x) for thousands of step, then go down very fast to 2.6 ~ 3.0.
Use smaller learning rate can help a little bit.

## Variations of FFTNet

### Radix-N FFTNet

Use the flag _--radixs_ to specify each layer's radix.

```
# a radix-4 FFTNet with 1024 receptive field
python train.py --radixs 4 4 4 4 4
```

The original FFtNet use Radix-2 structure. In my experiment, a radix-4 network can still achieved similar result, 
even radix-8, and by reduce the number of layers, it can run faster.

### Transposed FFTNet

Fig. 2 in the paper can be redraw as dilated structure with kernel size 2 (also means radix size 2).

![](images/fftnet_dilated.png)

If we draw all the lines;

![](images/fftnet_dilated2.png)

and transpose the the graph to let the arrows go backward, you'll find a WaveNet dilated structure.

![](images/fftnet_wavenet.png)

Add the flag __--transpose__, you can get a simplified version of WaveNet.
```
# a WaveNet-like structure model withou gated/residual/skip unit.
python train.py --transpose
```
In my experiment, the transposed models are more easy to train and have slightly lower training loss compare to FFTNet.
