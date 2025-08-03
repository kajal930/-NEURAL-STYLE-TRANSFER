import torch
import torchvision.transforms as T
from torchvision.models import vgg19
from PIL import Image
import matplotlib.pyplot as plt
import os
print("Current directory:", os.getcwd())

# Image load & preprocess
def load_image(path, max_size=300):
    img = Image.open(path).convert('RGB')
    transform = T.Compose([
        T.Resize((max_size, max_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

# De-normalize & show image
def show(tensor):
    image = tensor.clone().squeeze()
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    image = image + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    plt.imgshow(image.permute(1,2,0).clamp(0,1))
    plt.axis('off')
    plt.show()

# Load images
device = 'cuda' if torch.cuda.is_available() else 'cpu'
content = load_image(r'C:\Users\LENEVO\Desktop\test\VS CODE\PYTHON\krishna_1.jpg').to(device)
style = load_image(r'C:\Users\LENEVO\Desktop\test\VS CODE\PYTHON\krishna_2.jpg').to(device)

# Load model
vgg = vgg19(pretrained=True).features[:21].to(device).eval()

# Feature extraction
def get_features(x):
    features = []
    for layer in vgg:
        x = layer(x)
        features.append(x)
    return features

# Gram matrix
def gram_matrix(x):
    b, c, h, w = x.size()
    f = x.view(c, h * w)
    return torch.mm(f, f.t()) / (c * h * w)

# Loss & optimizer
target = content.clone().requires_grad_(True)
optimizer = torch.optim.Adam([target], lr=0.01)

for step in range(300):
    target_feats = get_features(target)
    content_feats = get_features(content)
    style_feats = get_features(style)

    content_loss = torch.mean((target_feats[10] - content_feats[10])**2)
    style_loss = 0
    for tf, sf in zip(target_feats, style_feats):
        style_loss += torch.mean((gram_matrix(tf) - gram_matrix(sf))**2)

    total_loss = 1e5 * style_loss + content_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f'Step {step} - Loss: {total_loss.item():.4f}')

# Show final output
show(target.cpu().detach())

