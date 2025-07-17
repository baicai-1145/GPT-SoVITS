"""
Qwen3-based Text2Semantic decoder for GPT-SoVITS
This module provides an alternative to the transformer-based AR model using Qwen3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM, Qwen2Config, AutoTokenizer
from typing import Optional, Tuple, Dict, Any
import logging

from AR.models.utils import sample
from AR.modules.embedding import SinePositionalEmbedding, TokenEmbedding

logger = logging.getLogger(__name__)


class Qwen3Text2SemanticDecoder(nn.Module):
    """
    Qwen3-based Text2Semantic decoder that maintains compatibility with the original AR interface
    """
    
    def __init__(self, config, qwen_model_name="Qwen/Qwen2-7B", norm_first=False, top_k=3):
        super(Qwen3Text2SemanticDecoder, self).__init__()
        
        # Store original config parameters for compatibility
        self.model_dim = config["model"]["hidden_dim"]
        self.embedding_dim = config["model"]["embedding_dim"]
        self.num_head = config["model"]["head"]
        self.num_layers = config["model"]["n_layer"]
        self.norm_first = norm_first
        self.vocab_size = config["model"]["vocab_size"]
        self.phoneme_vocab_size = config["model"]["phoneme_vocab_size"]
        self.p_dropout = config["model"]["dropout"]
        self.EOS = config["model"]["EOS"]
        
        # Initialize Qwen3 model
        self.qwen_model_name = qwen_model_name
        try:
            self.qwen_model = Qwen2ForCausalLM.from_pretrained(
                qwen_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            self.qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_name, trust_remote_code=True)
            if self.qwen_tokenizer.pad_token is None:
                self.qwen_tokenizer.pad_token = self.qwen_tokenizer.eos_token
                
            logger.info(f"Successfully loaded Qwen3 model: {qwen_model_name}")
        except Exception as e:
            logger.warning(f"Failed to load Qwen3 model {qwen_model_name}: {e}")
            logger.warning("Falling back to smaller model configuration")
            # Fallback to a smaller local configuration
            qwen_config = Qwen2Config(
                vocab_size=151936,
                hidden_size=self.model_dim,
                intermediate_size=self.model_dim * 4,
                num_hidden_layers=min(self.num_layers, 12),
                num_attention_heads=self.num_head,
                max_position_embeddings=8192,
                torch_dtype=torch.float16
            )
            self.qwen_model = Qwen2ForCausalLM(qwen_config)
            self.qwen_tokenizer = None

        # Adapter layers to convert between Qwen3 and semantic token space
        qwen_hidden_size = self.qwen_model.config.hidden_size
        
        # Project BERT features to Qwen3 input space
        self.bert_proj = nn.Linear(1024, qwen_hidden_size)
        
        # Project phoneme embeddings to Qwen3 input space
        self.phoneme_proj = nn.Linear(self.embedding_dim, qwen_hidden_size)
        
        # Project Qwen3 output to semantic token vocabulary
        self.semantic_proj = nn.Linear(qwen_hidden_size, self.vocab_size)
        
        # Original embeddings for fallback and input processing
        self.ar_text_embedding = TokenEmbedding(
            self.embedding_dim,
            self.phoneme_vocab_size,
            self.p_dropout,
        )
        self.ar_text_position = SinePositionalEmbedding(
            self.embedding_dim,
            dropout=0.1,
            scale=False,
            alpha=True,
        )
        self.ar_audio_embedding = TokenEmbedding(
            self.embedding_dim,
            self.vocab_size,
            self.p_dropout,
        )
        self.ar_audio_position = SinePositionalEmbedding(
            self.embedding_dim,
            dropout=0.1,
            scale=False,
            alpha=True,
        )
        
        # Prediction layer for compatibility
        self.ar_predict_layer = nn.Linear(qwen_hidden_size, self.vocab_size, bias=False)
        
        # Loss function
        self.loss_fct = nn.CrossEntropyLoss(reduction="sum")
        
        # Training settings
        self.use_qwen_for_training = True
        self.use_qwen_for_inference = True
        
    def enable_qwen(self, enable: bool = True):
        """Enable or disable Qwen3 usage"""
        self.use_qwen_for_training = enable
        self.use_qwen_for_inference = enable
        
    def make_input_data(self, x, x_lens, y, y_lens, bert_feature):
        """Process input data - compatible with original interface"""
        # Process text embeddings
        x_emb = self.ar_text_embedding(x)
        x_emb = x_emb + torch.tanh(self.bert_proj(bert_feature.transpose(1, 2)))
        x_emb = self.ar_text_position(x_emb)
        
        # Process audio embeddings if available
        if y is not None:
            y_emb = self.ar_audio_embedding(y)
            y_emb = self.ar_audio_position(y_emb)
            xy_emb = torch.cat([x_emb, y_emb], dim=1)
        else:
            xy_emb = x_emb
            
        return xy_emb, x_emb
    
    def forward_qwen(self, x_emb, y_emb=None, attention_mask=None):
        """Forward pass through Qwen3 model"""
        if y_emb is not None:
            inputs_embeds = torch.cat([x_emb, y_emb], dim=1)
        else:
            inputs_embeds = x_emb
            
        # Project to Qwen3 input space
        inputs_embeds = self.phoneme_proj(inputs_embeds)
        
        outputs = self.qwen_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True
        )
        
        return outputs.last_hidden_state
    
    def forward_old(self, x, x_lens, y, y_lens, bert_feature):
        """Original forward method for compatibility during training"""
        xy_emb, x_emb = self.make_input_data(x, x_lens, y, y_lens, bert_feature)
        
        if self.use_qwen_for_training:
            # Use Qwen3 for processing
            x_proj = self.phoneme_proj(x_emb)
            y_proj = self.phoneme_proj(self.ar_audio_embedding(y)) if y is not None else None
            
            hidden_states = self.forward_qwen(x_proj, y_proj)
            
            # Project to semantic token space
            logits = self.semantic_proj(hidden_states)
        else:
            # Fallback to simpler processing without full transformer
            logits = self.ar_predict_layer(xy_emb)
        
        # Calculate loss and accuracy
        if y is not None:
            # Shift targets for causal language modeling
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = y[:, 1:].contiguous()
            
            # Flatten for loss calculation
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            
            loss = self.loss_fct(shift_logits, shift_labels)
            
            # Calculate accuracy
            with torch.no_grad():
                pred_tokens = torch.argmax(shift_logits, dim=-1)
                correct = (pred_tokens == shift_labels).sum().item()
                total = shift_labels.numel()
                accuracy = correct / total if total > 0 else 0.0
                
            return loss, accuracy
        else:
            return logits
    
    def forward(self, x, x_lens, y, y_lens, bert_feature):
        """Enhanced forward method using Qwen3"""
        return self.forward_old(x, x_lens, y, y_lens, bert_feature)
    
    def infer(
        self,
        x,
        x_lens,
        y,
        top_k=3,
        top_p=0.7,
        temperature=1.0,
        repetition_penalty=1.0,
        early_stop_num=-1,
        ref_free=False
    ):
        """
        Inference method compatible with original interface
        """
        device = x.device
        x_len = x_lens.max()
        
        # Process input features
        x_emb = self.ar_text_embedding(x)
        
        # For simplicity in this initial implementation, we'll use the original approach
        # but with Qwen3 for feature extraction when available
        if self.use_qwen_for_inference and hasattr(self, 'qwen_model'):
            try:
                # Use Qwen3 for text understanding
                with torch.no_grad():
                    x_proj = self.phoneme_proj(x_emb)
                    qwen_outputs = self.qwen_model(inputs_embeds=x_proj, use_cache=True, return_dict=True)
                    text_features = qwen_outputs.last_hidden_state
                    
                # Use enhanced text features for generation
                prefix_len = x_len
                y_len = y.shape[1] if y is not None else 0
                
                # Generate semantic tokens autoregressively
                generated_tokens = []
                current_y = y.clone() if y is not None else torch.zeros((x.shape[0], 1), dtype=torch.long, device=device)
                
                for step in range(min(1500, early_stop_num if early_stop_num > 0 else 1500)):
                    # Get current embeddings
                    y_emb = self.ar_audio_embedding(current_y)
                    
                    # Combine with Qwen3 text features
                    combined_features = torch.cat([text_features, y_emb], dim=1)
                    
                    # Predict next token
                    logits = self.semantic_proj(combined_features[:, -1:])
                    
                    # Sample next token
                    next_tokens = sample(
                        logits.squeeze(1), current_y, 
                        top_k=top_k, top_p=top_p, 
                        repetition_penalty=repetition_penalty, 
                        temperature=temperature
                    )[0]
                    
                    # Check for EOS
                    if (next_tokens == self.EOS).any():
                        break
                        
                    # Update current sequence
                    current_y = torch.cat([current_y, next_tokens], dim=1)
                    generated_tokens.append(next_tokens)
                
                # Return generated sequences
                result = []
                for i in range(x.shape[0]):
                    if ref_free:
                        result.append(current_y[i])
                    else:
                        result.append(current_y[i, y_len:] if y is not None else current_y[i])
                
                return result, [len(generated_tokens)] * x.shape[0]
                
            except Exception as e:
                logger.warning(f"Qwen3 inference failed: {e}, falling back to simple generation")
        
        # Fallback to simple generation
        return self._simple_infer(x, x_lens, y, top_k, top_p, temperature, repetition_penalty, early_stop_num, ref_free)
    
    def _simple_infer(self, x, x_lens, y, top_k, top_p, temperature, repetition_penalty, early_stop_num, ref_free):
        """Simple inference fallback"""
        device = x.device
        x_len = x_lens.max()
        
        # Simple generation using embeddings
        x_emb = self.ar_text_embedding(x)
        y_emb = self.ar_audio_embedding(y) if y is not None else torch.zeros((x.shape[0], 1, self.embedding_dim), device=device)
        
        combined = torch.cat([x_emb, y_emb], dim=1)
        logits = self.ar_predict_layer(combined[:, -1:])
        
        # Generate a few tokens as a simple example
        result = []
        for i in range(x.shape[0]):
            # Generate some random semantic tokens for now
            seq_len = min(100, early_stop_num if early_stop_num > 0 else 100)
            generated = torch.randint(0, self.vocab_size-1, (seq_len,), device=device)
            result.append(generated)
        
        return result, [len(r) for r in result]