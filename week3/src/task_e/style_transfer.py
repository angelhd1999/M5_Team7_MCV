import os
from os import listdir
from os.path import isfile, join

import torch
import torchvision.models as models
from PIL import Image
from torchvision.utils import save_image

from config import RESULT_PATH, STYLE_PATH, CONTENT_PATH, NUM_STEPS, CONTENT_WEIGHT, STYLE_WEIGHT
from utils import image_loader, make_dirs, device, run_style_transfer, cnn_normalization_mean, cnn_normalization_std

# Generate a folder to save results
make_dirs(RESULT_PATH)

style_images = [f for f in listdir(STYLE_PATH) if isfile(join(STYLE_PATH, f))]
content_images = [f for f in listdir(CONTENT_PATH) if isfile(join(CONTENT_PATH, f))]

# sort list
style_images = sorted(style_images, key=lambda x: int(os.path.splitext(x)[0]))
content_images = sorted(content_images, key=lambda x: int(os.path.splitext(x)[0]))

cnn = models.vgg19(pretrained=True).features.to(device).eval()

for ii, (style, content) in enumerate(zip(style_images, content_images)):
    # relevant path
    style = join(STYLE_PATH, style)
    content = join(CONTENT_PATH, content)

    # print(style)
    # print(content)

    aux = Image.open(content)
    content_size = aux.size
    print(content_size)

    if min(content_size) > 1000:
        content_size = (800, 800)

    style_img = image_loader(style, min(content_size))
    content_img = image_loader(content, min(content_size))

    assert style_img.size() == content_img.size(), "style & content imgs should be same size"

    # input image as initializer
    input_img = content_img.clone()
    # uncomment & replace 4 white noise initializer

    """
    input_img = torch.randn(content_img.data.size(), device=device, requires_grad=False)
    # Gaussian smooth
    blur = cv2.GaussianBlur(input_img.detach().squeeze().cpu().numpy(), (3, 3), 0)
    blur = blur - blur.min()
    blur = blur / blur.max()
    blur = torch.from_numpy(blur).to(device).unsqueeze(0)
    """

    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img, num_steps=NUM_STEPS, print_step=100,
                                content_weight=CONTENT_WEIGHT, style_weight=STYLE_WEIGHT)

    # FIXME: maybe reshape before saving?
    # output = output.reshape(content_size)
    save_image(output, os.path.join(RESULT_PATH, f'{ii}.png'))
    torch.cuda.empty_cache()

print("PROCESS FINISHED")