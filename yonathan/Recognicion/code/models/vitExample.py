from transformers import ViTFeatureExtractor, ViTModel,ViTConfig
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

configuration = ViTConfig(hidden_size=768, num_hidden_layers=12, num_attention_heads=6, intermediate_size=3072,hidden_act='gelu',image_size=224, patch_size=16, num_labels=1000, classifier_dropout=0.1, attention_dropout=0.1, dropout=0.1, initializer_range=0.02, layer_norm_eps=1e-12, gradient_checkpointing=False, is_decoder=False, add_cross_attention=False, use_cache=True, model_type='vit',encoder_stride=1)


feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224')

inputs = feature_extractor(images=image, return_tensors="pt")
# the FeatureExtractor is used to preprocess the image, resize it to 224x224 and normalize it with the mean and standard deviation of the ImageNet dataset
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state