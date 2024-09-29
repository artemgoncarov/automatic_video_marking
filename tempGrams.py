from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Загрузка предобученной модели ResNet50
model = models.resnet50(pretrained=True)
model.eval()

# Функция для получения градиентов
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        # Регистрируем хуки для слоев
        self.hook_handles = []
        self.hook_handles.append(target_layer.register_forward_hook(self.save_activations))
        self.hook_handles.append(target_layer.register_backward_hook(self.save_gradients))

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, input_image, class_idx):
        # Прямой проход
        output = self.model(input_image)
        self.model.zero_grad()
        output[0, class_idx].backward()  # Вычисляем градиент для целевого класса
        gradients = self.gradients.data.numpy()[0]
        activations = self.activations.data.numpy()[0]

        # Усредняем градиенты
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_image.size(2), input_image.size(3)))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

# Функция для подготовки изображения
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0)
    return image

# Функция для применения Grad-CAM к изображениям из папки и их визуализации
def apply_grad_cam_to_folder(filename, class_idx=243):
    # Инициализируем Grad-CAM для последнего свёрточного слоя ResNet50
    grad_cam = GradCAM(model, model.layer4[2].conv3)  # Последний свёрточный слой ResNet50

    input_image = preprocess_image(filename)

    # Получаем тепловую карту
    cam = grad_cam(input_image, class_idx)

    # Визуализация тепловой карты
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    img = cv2.imread(filename)
    img = cv2.resize(img, (224, 224))
    heatmap = np.float32(heatmap) / 255
    overlay = heatmap + np.float32(img) / 255
    overlay = overlay / np.max(overlay)

    # Сохраняем результат
    output_path = os.path.join('static/uploads', 'image.jpeg')
    plt.imsave(output_path, overlay)


# Пример использования: применяем Grad-CAM ко всем изображениям в папке и визуализируем их
# input_folder = 'frame.jpeg'
# class_idx = 243  # Можно изменить целевой класс в зависимости от задачи

# apply_grad_cam_to_folder(input_folder, class_idx)