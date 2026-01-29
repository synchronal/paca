Paca
====

Helpers for interacting with llama.cpp.

## Download

Llama.cpp has CLI options that automatically download models from huggingface,
but does not have clear, documented facilities for doing so without loading
them. This can be problematic for cases where model files must be downloaded
via computers without the available RAM to run them, for example when the
llama.cpp server lives on a network with slow downlink speeds.

Paca provides a `download` subcommand for this purpose.

``` shell
paca download unsloth/GLM-4.7-GGUF:BF16
```
