paca-cli
====

The CLI for `paca`. This CLI provides subcommands for interacting with
the huggingface cache, specifically designed to help manage local models for
llama.cpp. While it implements some redundant commands to the huggingface CLI,
it is primarily focused on downloading models and managing the local cache
directory.

## Subcommands

### Clean

Removes any stray partial downloads that may have snuck into the cache directory.

### Download

Llama.cpp has CLI options that automatically download models from huggingface,
but does not have clear, documented facilities for doing so without loading
them. This can be problematic for cases where model files must be downloaded
via computers without the available RAM to run them, for example when the
llama.cpp server lives on a network with slow downlink speeds.

Paca provides a `download` subcommand for this purpose.

``` shell
paca download unsloth/GLM-4.7-GGUF:BF16
paca dl unsloth/GLM-4.7-GGUF:BF16
```

### List

List all downloaded models.

``` shell
paca list
paca ls
```

### Outdated

Verify all downloaded models against their current versions in huggingface.

``` shell
paca outdated
paca o
```

### Remove

Remove a local model

``` shell
paca remove unsloth/GLM-4.7-GGUF:BF16
paca rm unsloth/GLM-4.7-GGUF:BF16
```
