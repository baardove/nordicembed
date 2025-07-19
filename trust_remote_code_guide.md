# Trust Remote Code Guide

## Overview

Some models (like `norbert3-base` and `norbert3-large`) contain custom code that needs to be executed for the model to work properly. NoEmbed now supports these models with proper security controls.

## Security Notice

**⚠️ Warning**: Enabling `trust_remote_code` means executing code from the model repository. Only use models from trusted sources.

## Configuration

### Global Setting

Control whether to allow models with custom code via the `.env` file:

```env
# Security
ALLOW_TRUST_REMOTE_CODE=true  # Set to false to disable
```

- `true` (default): Allows models that require custom code
- `false`: Blocks models that require custom code

### Models Requiring Trust Remote Code

The following models require `trust_remote_code=True`:

| Model | Repository | Description |
|-------|------------|-------------|
| `norbert3-base` | ltg/norbert3-base | Norwegian BERT v3 base model |
| `norbert3-large` | ltg/norbert3-large | Norwegian BERT v3 large model |

## How It Works

1. When loading a model that requires custom code:
   - The service checks if `ALLOW_TRUST_REMOTE_CODE=true`
   - If allowed, it downloads and executes the custom code
   - A warning is logged about the custom code execution

2. Security checks:
   - Global setting must allow trust_remote_code
   - Each model explicitly declares if it needs custom code
   - Warnings are logged when custom code is used

## Error Messages

### "Model requires trust_remote_code=True"

If you see this error:
```
The repository for ltg/norbert3-large contains custom code which must be executed to correctly load the model. 
Please pass the argument `trust_remote_code=True` to allow custom code to be run.
```

**Solution**: The model is already configured to use trust_remote_code. Ensure `ALLOW_TRUST_REMOTE_CODE=true` in your `.env` file.

### "ALLOW_TRUST_REMOTE_CODE is disabled"

If you see:
```
Model norbert3-large requires trust_remote_code=True, but ALLOW_TRUST_REMOTE_CODE is disabled. 
Set ALLOW_TRUST_REMOTE_CODE=true in your .env file to enable.
```

**Solution**: Add or update in your `.env` file:
```env
ALLOW_TRUST_REMOTE_CODE=true
```

## Best Practices

1. **Review Model Sources**: Only use models from trusted organizations
2. **Pin Model Versions**: Consider pinning specific model revisions for production
3. **Monitor Logs**: Check logs for warnings about custom code execution
4. **Disable When Not Needed**: Set `ALLOW_TRUST_REMOTE_CODE=false` if you don't need these models

## Testing

Test a model that requires custom code:
```bash
curl -X POST http://localhost:7000/api/test-embed \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Test norbert3 model"],
    "model": "norbert3-large",
    "pooling_strategy": "mean"
  }'
```

## Log Messages

When loading models with custom code, you'll see:
```
WARNING - Model ltg/norbert3-large requires trust_remote_code=True. This will execute custom code from the model repository.
```

And from HuggingFace:
```
A new version of the following files was downloaded from https://huggingface.co/ltg/norbert3-large:
- configuration_norbert.py
- modeling_norbert.py
Make sure to double-check they do not contain any added malicious code.
```