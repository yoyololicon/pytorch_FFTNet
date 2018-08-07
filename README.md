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
    --preprocess
    --wav_dir your_downloaded_wav_dir
    --data_dir preprocessed_feature_dir
    --model_file saved_model_name
    
```

[FFTNet_generator](FFTNet_generator.py) and [FFTNet_vocoder](FFTNet_vocoder.py) are two files I used to test the model 
workability using torchaudio yesno dataset.