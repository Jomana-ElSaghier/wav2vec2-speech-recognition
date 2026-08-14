import torch.nn as nn
from transformers import Wav2Vec2Model
 
 
class Wav2VecClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
 
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
 
        for param in self.wav2vec.feature_extractor.parameters():
            param.requires_grad = False
 
        self.classifier = nn.Linear(768, num_classes)
 
    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec(
            input_values=input_values,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state
        pooled = hidden_states.mean(dim=1)
        logits = self.classifier(pooled)
        return logits
 
 
def build_model(num_classes=10):
    return Wav2VecClassifier(num_classes=num_classes)
 