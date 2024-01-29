import numpy as np
from tqdm import tqdm
import torch
from imageio import imread, imwrite
from path import Path
import os
from PIL import Image

from config import get_opts, get_training_size

from SC_Depth import SC_Depth
from SC_DepthV2 import SC_DepthV2
from SC_DepthV3 import SC_DepthV3

import datasets.custom_transforms as custom_transforms

from visualization import *


@torch.no_grad()
def main():
    hparams = get_opts()

    if hparams.model_version == 'v1':
        system = SC_Depth(hparams)
    elif hparams.model_version == 'v2':
        system = SC_DepthV2(hparams)
    elif hparams.model_version == 'v3':
        system = SC_DepthV3(hparams)

    system = system.load_from_checkpoint(hparams.ckpt_path, strict=False)

    model = system.depth_net
    # model.cuda()
    model.eval()

    # training size
    training_size = get_training_size(hparams.dataset_name)

    # normalization
    inference_transform = custom_transforms.Compose([
        custom_transforms.RescaleTo(training_size),
        custom_transforms.ArrayToTensor(),
        custom_transforms.Normalize()]
    )

    input_dir = Path(hparams.input_dir)
    output_dir = Path(hparams.output_dir) / \
        'model_{}'.format(hparams.model_version)
    output_dir.makedirs_p()

    if hparams.save_vis:
        (output_dir/'vis').makedirs_p()

    if hparams.save_depth:
        (output_dir/'depth').makedirs_p()

    image_files = sum([(input_dir).files('*.{}'.format(ext))
                      for ext in ['jpg', 'png']], [])
    image_files = sorted(image_files)

    print('{} images for inference'.format(len(image_files)))

    for i, img_file in enumerate(tqdm(image_files)):

        filename = os.path.splitext(os.path.basename(img_file))[0]

        img = Image.open(img_file).convert('RGB')
        original_size = img.size  # save original size
        img = np.array(img).astype(np.float32)
        tensor_img = inference_transform([img])[0][0].unsqueeze(0) #.cuda()
        pred_depth = model(tensor_img)

        # resize depth map to original size
        pred_depth = pred_depth.squeeze().cpu().numpy()
        pred_depth_resized = cv2.resize(pred_depth, original_size, interpolation=cv2.INTER_LINEAR)

        if hparams.save_vis:
            vis = visualize_depth(torch.from_numpy(pred_depth_resized), output_size=original_size)
            vis = vis.permute(1, 2, 0).numpy() * 255
            imwrite(output_dir/'vis/{}.jpg'.format(filename), vis.astype(np.uint8))

        if hparams.save_depth:
            np.save(output_dir/'depth/{}.npy'.format(filename), pred_depth_resized)


if __name__ == '__main__':
    main()
