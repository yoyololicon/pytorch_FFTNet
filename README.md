This is a pytorch implementation of FFTNet described [here](http://gfx.cs.princeton.edu/pubs/Jin_2018_FAR/).
Work in progress.

## Quick Start

1. Install requirements
```
pip install -r requirements.txt
```

2. Download [CMU_ARCTIC](http://festvox.org/cmu_arctic/) dataset.

3. Train the model and save. Raise the flag _--preprocess_ when execute the first time.

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

## TODO

- [x] Add generation/decode on test dataset.
- [ ] Implement voiced/unvoiced conditional sampling.
- [ ] Post-synthesis denoising.
- [ ] Add checkpoints function in training.

## Notes

* I combine two 1x1 convolution kernel to one 1x2 dilated kernel.
This can remove redundant bias parameters and accelerate total speed.
* Some details are differ from the paper.