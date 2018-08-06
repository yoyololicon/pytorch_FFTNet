import tensorflow as tf

# Default hyperparameters:
hparams = tf.contrib.training.HParams(

    # Audio:
    n_mfcc=25,
    minf0=40,
    maxf0=500,
    sample_rate=16000,
    winlen=0.025,
    winstep=0.01,
    preemphasis=0.97,
    noise_injecting=True,
    interp_method='repeat',

    # Training:
    use_cuda=True,
    use_local_condition=True,
    batch_size=5,
    sample_size=5000,
    learning_rate=2e-4,
    training_steps=10000,
    checkpoint_interval=5000,

    # Model
    radixs=[2] * 11,
    fft_channels=256,
    quantization_channels=256,

    # Test
    c=2
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)


if __name__ == '__main__':
    print(hparams_debug_string())
