from transformers import pipeline
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
print(captioner("/home/meoconlonton/AI-chat-assistant/test_img/haerin.jpg")[0]["generated_text"])
