from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

# Input Image
filename = "/Users/Illusion/Documents/rk_face.jpg"

input_img = Image.open(filename)
#input_img = input_img.resize((INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT), Image.LANCZOS)

#print list(input_img.getdata())

normalized_img = transform(input_img).unsqueeze(0)

(1L, 3L, 606L, 472L)

print normalized_img[0][2][0][0]
print normalized_img[0][2][1][1]
print normalized_img[0][2][2][2]
print normalized_img[0][2][3][3]
#print normalized_img.shape