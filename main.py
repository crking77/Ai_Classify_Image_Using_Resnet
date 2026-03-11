
import numpy as np
import torch
from torchvision.models import mobilenet_v2,MobileNet_V2_Weights
from torchvision import models, transforms
import streamlit as st
from PIL import Image


weights = MobileNet_V2_Weights.DEFAULT
categories = weights.meta["categories"]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)
def preprocess_image(img):
    return transform(img)


def main():
    st.title("AI classifier")
    @st.cache_resource
    def load_cache_model():
        load_model = models.mobilenet_v2(weights = MobileNet_V2_Weights.IMAGENET1K_V1)
        load_model.eval()
        return load_model
    model = load_cache_model()
    file_uploader = st.file_uploader(label = "Choose image for predict",type=["png","jpg"])
    if file_uploader is not None:
        st.image(file_uploader,"This is image upload!")
        btn = st.button("Classify image")
        if btn:
            with st.spinner("Analyzing image..."):
                image = Image.open(file_uploader).convert("RGB")
                image = transform(image)
                image = image.unsqueeze(0)
                with torch.no_grad():
                    result = model(image)
                prob = torch.nn.functional.softmax(result, dim=1)
                print(prob)
                top_prob, top_catid = torch.topk(prob, 3)
                for i in range(3):
                    label = categories[top_catid[0][i]]
                    confidence = top_prob[0][i].item()
                    st.write(f"{label}: {confidence:.2%}")
if __name__ == "__main__":
    main()
