#!/usr/bin/env python3
"""
Demo script showing how to use Qwen3 integration in GPT-SoVITS
"""

import os
import sys
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add GPT_SoVITS to path
sys.path.insert(0, 'GPT_SoVITS')

def demo_qwen3_config():
    """Demo 1: Basic configuration setup"""
    print("Demo 1: Basic Qwen3 Configuration")
    print("-" * 40)
    
    # Set environment variables
    os.environ["use_qwen3"] = "True"
    os.environ["qwen3_model_name"] = "Qwen/Qwen2-1.5B"
    
    try:
        import config
        print(f"✓ use_qwen3: {getattr(config, 'use_qwen3', 'Not found')}")
        print(f"✓ qwen3_model_name: {getattr(config, 'qwen3_model_name', 'Not found')}")
        
        # Create config instance
        cfg = config.Config()
        print(f"✓ Config instance created with Qwen3 settings")
        print(f"  - use_qwen3: {getattr(cfg, 'use_qwen3', 'Not found')}")
        print(f"  - qwen3_model_name: {getattr(cfg, 'qwen3_model_name', 'Not found')}")
        
    except Exception as e:
        print(f"❌ Config demo failed: {e}")
    
    print()

def demo_qwen3_model():
    """Demo 2: Qwen3 model instantiation"""
    print("Demo 2: Qwen3 Model Instantiation")
    print("-" * 40)
    
    try:
        from AR.models.qwen3_t2s_model import Qwen3Text2SemanticDecoder
        print("✓ Qwen3Text2SemanticDecoder imported successfully")
        
        # Create a minimal config for testing
        test_config = {
            "model": {
                "hidden_dim": 512,
                "embedding_dim": 512,
                "head": 8,
                "n_layer": 12,
                "vocab_size": 1025,
                "phoneme_vocab_size": 512,
                "dropout": 0.0,
                "EOS": 1024
            }
        }
        
        # Try to create model instance (may fail without dependencies, which is expected)
        try:
            model = Qwen3Text2SemanticDecoder(test_config, qwen_model_name="Qwen/Qwen2-0.5B")
            print("✓ Qwen3Text2SemanticDecoder instance created")
            print(f"  - Model dimension: {model.model_dim}")
            print(f"  - Vocabulary size: {model.vocab_size}")
            print(f"  - Using Qwen3: {getattr(model, 'use_qwen_for_inference', 'Unknown')}")
        except Exception as e:
            print(f"⚠️  Model instantiation failed (expected without dependencies): {type(e).__name__}")
            print("   This is normal if transformers/torch are not installed")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
    
    print()

def demo_tts_config():
    """Demo 3: TTS configuration with Qwen3"""
    print("Demo 3: TTS Configuration with Qwen3")
    print("-" * 40)
    
    try:
        from TTS_infer_pack.TTS import TTS_Config
        
        # Create TTS config with Qwen3 settings
        config_dict = {
            "custom": {
                "device": "cpu",
                "is_half": False,
                "version": "v2",
                "use_qwen3": True,
                "qwen3_model_name": "Qwen/Qwen2-1.5B",
                "qwen3_enable_training": False,
                "qwen3_enable_inference": True,
                # Add minimum required paths (they don't need to exist for this demo)
                "t2s_weights_path": "dummy_path.ckpt",
                "vits_weights_path": "dummy_path.pth", 
                "bert_base_path": "dummy_path",
                "cnhuhbert_base_path": "dummy_path"
            }
        }
        
        tts_config = TTS_Config(config_dict)
        print("✓ TTS_Config created with Qwen3 settings")
        print(f"  - use_qwen3: {getattr(tts_config, 'use_qwen3', 'Not found')}")
        print(f"  - qwen3_model_name: {getattr(tts_config, 'qwen3_model_name', 'Not found')}")
        print(f"  - device: {tts_config.device}")
        
    except Exception as e:
        print(f"❌ TTS config demo failed: {e}")
    
    print()

def demo_lightning_module():
    """Demo 4: Lightning module with Qwen3"""
    print("Demo 4: Lightning Module with Qwen3")
    print("-" * 40)
    
    try:
        from AR.models.t2s_lightning_module import Text2SemanticLightningModule
        
        # Create a test config
        test_config = {
            "model": {
                "hidden_dim": 512,
                "embedding_dim": 512,
                "head": 8,
                "n_layer": 12,
                "vocab_size": 1025,
                "phoneme_vocab_size": 512,
                "dropout": 0.0,
                "EOS": 1024
            },
            "qwen3_model_name": "Qwen/Qwen2-0.5B"
        }
        
        try:
            # Test without Qwen3
            lightning_module = Text2SemanticLightningModule(test_config, "/tmp", is_train=False, use_qwen3=False)
            print("✓ Lightning module created (original AR model)")
            
            # Test with Qwen3 (may fail without dependencies)
            try:
                lightning_module_qwen3 = Text2SemanticLightningModule(test_config, "/tmp", is_train=False, use_qwen3=True)
                print("✓ Lightning module created (Qwen3 model)")
            except Exception as e:
                print(f"⚠️  Qwen3 Lightning module failed (expected): {type(e).__name__}")
                
        except Exception as e:
            print(f"❌ Lightning module creation failed: {e}")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
    
    print()

def main():
    """Run all demos"""
    print("GPT-SoVITS Qwen3 Integration Demo")
    print("=" * 50)
    print("This demo shows the Qwen3 integration features.")
    print("Some features may not work without full dependencies installed.")
    print()
    
    demos = [
        demo_qwen3_config,
        demo_qwen3_model,
        demo_tts_config,
        demo_lightning_module
    ]
    
    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            print()
    
    print("Demo completed!")
    print()
    print("To fully test the integration:")
    print("1. Install dependencies: pip install torch transformers qwen-vl")
    print("2. Run the WebUI: python webui.py")
    print("3. Enable Qwen3 in the interface")
    print("4. Try text-to-speech generation")

if __name__ == "__main__":
    main()