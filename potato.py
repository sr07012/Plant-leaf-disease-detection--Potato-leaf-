import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import io
import tensorflow as tf
import keras


model_weights_file = 'Image_classify.h5'  
classes = ['EarlyBlight', 'Healthy', 'LateBlight', 'Invalid']  


model = keras.models.load_model(model_weights_file)


disease_explanations = {
    'EarlyBlight': "Early blight, caused by the fungus Alternaria solani, commonly affects potato plants, manifesting as brown spots on leaves and stems. This disease can lead to reduced yield and quality of potato crops if not managed properly.",
    'Healthy': "The plant appears to be in a healthy condition, exhibiting no visible signs of disease or stress. It is crucial to maintain the health of potato plants through proper cultivation practices and disease management strategies.",
    'LateBlight': "Late blight, caused by the pathogen Phytophthora infestans, is a destructive disease affecting potato plants worldwide. It results in dark, water-soaked lesions on leaves, leading to rapid defoliation and crop loss if left untreated.",
    'Invalid': "This classification indicates that the system was unable to confidently classify the disease based on the provided image. It could be due to various factors such as image quality, lighting conditions, or the presence of multiple diseases. Further assessment or a different image may be necessary for accurate diagnosis."
}


def preprocess_image(img):
    
    img = img.convert("RGBA")
    
    
    new_img = Image.new("RGBA", img.size, "WHITE")
    
    
    new_img.paste(img, (0, 0), img)
    
    
    new_img = new_img.convert("RGB")
    
    return new_img


def classify_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        
        
        image = preprocess_image(image)
        
        
        image = image.resize((256, 256))
        
        
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)

        
        image = tf.keras.preprocessing.image.load_img(img_bytes, target_size=(256, 256))
        
        
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        
    
        img_array = np.expand_dims(img_array, axis=0)
        
        
        prediction = model.predict(img_array)
        
        
        predicted_class_index = np.argmax(prediction)
        

        predicted_class_label = classes[predicted_class_index]
        
        
        explanation = disease_explanations.get(predicted_class_label, 'Explanation not available.')
        
        
        result_label.config(text=f"Predicted Class: {predicted_class_label}")
        
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo
        
        result_label1.config(text=f"Explanation: {explanation}")
        
        


root = tk.Tk()
root.title("Image Classification")


root.geometry("700x550")


root.configure(bg="#f8f9fa")


padx = 20
pady = 10


title_label = tk.Label(root, text="Disease Identification of Potato Leaf", font=("Arial", 16), bg="#f8f9fa")
title_label.pack(pady=(pady, 0))


select_button = tk.Button(root, text="Select Image", command=classify_image, font=("Arial", 12), bg="#28a745", fg="#fff")
select_button.pack(pady=pady)


result_label = tk.Label(root, text="", font=("Arial", 16), bg="#f8f9fa", justify="left", wraplength=480)
result_label.pack(pady=pady)

result_label1 = tk.Label(root, text="", font=("Arial", 12), bg="#f8f9fa", justify="left", wraplength=480)
result_label1.pack(pady=pady)

image_label = tk.Label(root, bg="#fff")
image_label.pack(pady=pady)


root.mainloop()
