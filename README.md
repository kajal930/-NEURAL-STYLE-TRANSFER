# -NEURAL-STYLE-TRANSFER## 🔧 Project Overview – Task 3: Neural Style Transfer

This project implements *Neural Style Transfer (NST)* — a deep learning technique that combines the *content of one image* with the *style of another* to create a new, stylized output. For example, applying the brushstroke style of a painting onto a photo of Krishna Bhagwan.

---

## 🧠 How It Was Built

### 1. 🔍 Pre-trained VGG19 Model
- We use *VGG19*, a deep convolutional neural network pre-trained on the ImageNet dataset.
- The model is used *only for feature extraction* — not for classification.

### 2. 🖼 Image Processing
- Content image and style image are loaded using *PIL (Python Imaging Library)*.
- Images are resized, normalized, and converted into PyTorch tensors.

### 3. 🎨 Feature Extraction
- From VGG19, we extract:
  - *Content Features* from deeper layers (e.g., conv4_2)
  - *Style Features* from multiple layers (e.g., conv1_1, conv2_1, etc.)
- We use a *Gram Matrix* to represent style features (texture patterns).

### 4. 📉 Loss Functions
- *Content Loss*: Measures how much the generated image differs from the content image.
- *Style Loss*: Measures how much the style (texture/color) differs from the style image.
- A *total loss* is calculated as a weighted sum of both.

### 5. ⚙️ Optimization
- The generated image is initialized as a copy of the content image.
- We use *gradient descent (Adam optimizer)* to iteratively update the image.
- After several iterations, the image learns to look like the content image in the style of the painting.

---

## 🧰 Libraries Used

| Library       | Purpose                                  |
|---------------|-------------------------------------------|
| torch       | Deep learning framework (PyTorch)         |
| torchvision | Pre-trained models and image transforms   |
| PIL         | Image loading and preprocessing           |
| matplotlib  | Display and save the output image         |

---

## 🖼 Input & Output

| Content Image       | Style Image         | Output Image         |
|---------------------|---------------------|----------------------|
| krishna.jpg         | painting.jpg        | output.jpg           |

---

## ✅ Final Output

A stylized image that *retains the subject and layout* of the original photo, but is rendered in the *artistic style* of the chosen painting.

---

## 📁 Folder Structure
