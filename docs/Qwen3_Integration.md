# Qwen3 Integration for GPT-SoVITS

This document describes the Qwen3 integration that enhances GPT-SoVITS with improved text understanding capabilities.

## Overview

The Qwen3 integration provides an optional replacement for the original AR (Autoregressive) transformer model in GPT-SoVITS. By leveraging Qwen3's advanced language understanding capabilities, users can potentially achieve better text-to-semantic token generation, especially for complex or nuanced text inputs.

## Features

- **Optional Integration**: Qwen3 can be enabled/disabled without affecting existing workflows
- **Multiple Model Sizes**: Support for different Qwen3 model variants (7B, 1.5B, 0.5B)
- **Backward Compatibility**: Full compatibility with existing GPT-SoVITS models and weights
- **Real-time Switching**: Change between original AR model and Qwen3 in the WebUI
- **Fallback Support**: Automatic fallback to original model if Qwen3 fails

## Installation

1. Install the additional dependencies:
```bash
pip install qwen-vl transformers>=4.43
```

2. The integration is already included in the latest GPT-SoVITS code.

## Usage

### WebUI Interface

1. **Launch GPT-SoVITS WebUI**:
```bash
python webui.py
```

2. **Enable Qwen3**:
   - In the model selection section, you'll see a checkbox labeled "使用Qwen3增强文本理解 (Use Qwen3 for enhanced text understanding)"
   - Check this box to enable Qwen3
   - Select your preferred Qwen3 model size from the dropdown (appears when enabled)

3. **Model Options**:
   - `Qwen/Qwen2-7B`: Best quality, higher resource usage
   - `Qwen/Qwen2-1.5B`: Balanced performance and speed
   - `Qwen/Qwen2-0.5B`: Fastest, lower resource usage

### Environment Variables

You can also configure Qwen3 using environment variables:

```bash
export use_qwen3=True
export qwen3_model_name="Qwen/Qwen2-1.5B"
export qwen3_enable_training=False
export qwen3_enable_inference=True
```

### API Usage

For programmatic use:

```python
from TTS_infer_pack.TTS import TTS, TTS_Config

# Create TTS config with Qwen3 enabled
config = TTS_Config()
config.use_qwen3 = True
config.qwen3_model_name = "Qwen/Qwen2-7B"

# Initialize TTS
tts = TTS(config)

# The TTS will now use Qwen3 for text processing
```

## Configuration Options

### Basic Settings

- `use_qwen3`: Enable/disable Qwen3 (default: False)
- `qwen3_model_name`: Qwen3 model to use (default: "Qwen/Qwen2-7B")
- `qwen3_enable_training`: Use Qwen3 during training (default: False)
- `qwen3_enable_inference`: Use Qwen3 during inference (default: True)

### Model Variants

| Model | Parameters | Memory Usage | Speed | Quality |
|-------|------------|--------------|-------|---------|
| Qwen/Qwen2-0.5B | 0.5B | ~2GB | Fast | Good |
| Qwen/Qwen2-1.5B | 1.5B | ~4GB | Medium | Better |
| Qwen/Qwen2-7B | 7B | ~14GB | Slow | Best |

## How It Works

1. **Text Processing**: When enabled, Qwen3 processes the input text to extract enhanced semantic representations
2. **Adapter Layer**: Custom adapter layers convert Qwen3's output to the format expected by SoVITS
3. **Semantic Generation**: The enhanced text understanding is used to generate better semantic tokens
4. **Fallback**: If Qwen3 fails, the system automatically falls back to the original AR model

## Benefits

- **Improved Text Understanding**: Better handling of complex sentences, context, and nuances
- **Enhanced Multilingual Support**: Leverage Qwen3's multilingual capabilities
- **Better Context Awareness**: Improved understanding of text context and meaning
- **Flexible Usage**: Can be enabled/disabled based on specific needs

## Limitations

- **Resource Usage**: Qwen3 models require additional GPU memory and computation
- **Inference Speed**: May be slower than the original AR model, especially for larger Qwen3 variants
- **Model Download**: First use requires downloading Qwen3 model weights from Hugging Face

## Troubleshooting

### Common Issues

1. **OutOfMemoryError**: Try using a smaller Qwen3 model (0.5B or 1.5B)
2. **Model Download Fails**: Ensure internet connection and Hugging Face access
3. **Slow Inference**: Use smaller models or disable Qwen3 for real-time applications

### Fallback Behavior

If Qwen3 fails to load or encounters errors:
- The system automatically falls back to the original AR model
- A warning message is displayed
- Inference continues with the original functionality

## Performance Tips

1. **Model Selection**: Choose the appropriate model size based on your hardware
2. **Batch Processing**: Use batch inference for better throughput with Qwen3
3. **Mixed Usage**: Enable Qwen3 only for complex texts that benefit from enhanced understanding

## Future Improvements

- Support for more Qwen3 model variants
- Optimized inference pipeline for better speed
- Fine-tuning capabilities for domain-specific tasks
- Integration with other advanced language models

## Contributing

To contribute to the Qwen3 integration:

1. The main implementation is in `GPT_SoVITS/AR/models/qwen3_t2s_model.py`
2. Configuration is handled in `config.py` and `TTS_infer_pack/TTS.py`
3. WebUI integration is in `inference_webui.py`

## License

This integration follows the same MIT license as the main GPT-SoVITS project.