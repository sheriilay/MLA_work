import torch
import torchvision
import torchvision.transforms as transforms
import PIL.Image as Image

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'nothing', 'space']

model = torch.load('model.pth')
mean = [0.4725, 0.4614, 0.4750]
std = [0.1818, 0.2227, 0.2390]

transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(), transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))])

def classify(model, image_transform, image_path, classes):
    model = model.eval()
    image = Image.open(image_path)
    image = image_transform(image).float()
    image = image.unsqueeze(0)
    
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    print(classes[predicted.item()])
    
classify(model, transforms, 'photo1707466703.jpeg', classes)