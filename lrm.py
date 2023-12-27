from .OpenLRM.lrm.inferrer import LRMInferrer
from diffusers import AutoPipelineForText2Image
import torch
import cv2
import rembg
from trimesh.sample import sample_surface

from dataclasses import dataclass, field
from typing import List
from PIL import Image
import os

import numpy as np
import threestudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from threestudio.utils.base import BaseObject
from threestudio.utils.typing import *


@threestudio.register("lrm-guidance")
class lrm_Guidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        model_name: str = "lrm-base-obj-v1"
        sd_model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"
        mode: str = "text2image" # image2image or text2image
        skip: int = 4
        source_image: str = "custom/threestudio-lrm/tmp.jpg"  # Add default path or leave empty
        dump_path: str = "custom/threestudio-lrm/OpenLRM/"  # Add default path or leave empty
        source_size: int = -1  # Default value or specify a size
        render_size: int = -1  # Default value or specify a size
        mesh_size: int = 384  # Default value or specify a size
        export_video: bool = False  # Default value
        export_mesh: bool = True  # Default value

    cfg: Config

    def configure(self) -> None:
        pass
    
    def densify(self, factor=2):
        pass

    def remove_background(self, image_path, output_size=512, border_ratio=0.2, recenter=True, model='u2net'):
        """
        Removes the background from an image.

        Args:
            image_path (str): Path to the image file.
            output_size (int): The resolution of the output image.
            border_ratio (float): Border ratio for the output image.
            recenter (bool): Whether to recenter the image after background removal.
            model (str): The model used for background removal.

        Returns:
            str: The path to the processed image.
        """
        threestudio.info(f"Removing Background ...")
        
        session = rembg.new_session(model_name=model)

        # Processing the image
        out_base = os.path.basename(image_path).split('.')[0]
        out_dir = os.path.dirname(image_path)
        out_rgba = os.path.join(out_dir, out_base + '_rgba.png')

        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        # Remove background
        carved_image = rembg.remove(image, session=session)  # [H, W, 4]
        mask = carved_image[..., -1] > 0

        # Recenter image if required
        if recenter:
            final_rgba = np.zeros((output_size, output_size, 4), dtype=np.uint8)
            
            coords = np.nonzero(mask)
            x_min, x_max = coords[0].min(), coords[0].max()
            y_min, y_max = coords[1].min(), coords[1].max()
            h, w = x_max - x_min, y_max - y_min
            desired_size = int(output_size * (1 - border_ratio))
            scale = desired_size / max(h, w)
            h2, w2 = int(h * scale), int(w * scale)
            x2_min, y2_min = (output_size - h2) // 2, (output_size - w2) // 2
            final_rgba[x2_min:x2_min+h2, y2_min:y2_min+w2] = cv2.resize(carved_image[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)
        else:
            final_rgba = carved_image
        
        # Save the processed image
        cv2.imwrite(out_rgba, final_rgba)
        return out_rgba
        
    def __call__(
        self,
        prompt,
        **kwargs,
    ):
        """
        Executes the lrm_Guidance object.

        Args:
            prompt: The input prompt.

        Returns:
            tuple: A tuple containing the coordinates and RGB values.
        """
        threestudio.info(f"Loading lrm guidance ...")

        if self.cfg.mode == "text2image":
            pipe = AutoPipelineForText2Image.from_pretrained(
                self.cfg.sd_model_name, 
                torch_dtype=torch.float16, variant="fp16"
            ).to("cuda")

            image = pipe(prompt).images[0]
            image = image.save(self.cfg.source_image)
        self.cfg.source_image = self.remove_background(self.cfg.source_image)
            
        with LRMInferrer(model_name=self.cfg.model_name) as inferrer:
            source_image_size = self.cfg.source_size if self.cfg.source_size > 0 else inferrer.infer_kwargs['source_size']

            image = torch.tensor(np.array(Image.open(self.cfg.source_image))).permute(2, 0, 1).unsqueeze(0) / 255.0
            # if RGBA, blend to RGB
            if image.shape[1] == 4:
                image = image[:, :3, ...] * image[:, 3:, ...] + (1 - image[:, 3:, ...])
            image = torch.nn.functional.interpolate(image, size=(source_image_size, source_image_size), mode='bicubic', align_corners=True)
            image = torch.clamp(image, 0, 1)
            results = inferrer.infer_single(
                image.to(self.device),
                render_size=self.cfg.render_size if self.cfg.render_size > 0 else inferrer.infer_kwargs['render_size'],
                mesh_size=self.cfg.mesh_size,
                export_video=self.cfg.export_video,
                export_mesh=self.cfg.export_mesh,
            )

        # Extract mesh and convert to point cloud
        mesh = results['mesh']
        # mesh.export(os.path.join(self.cfg.dump_path, f'tmp.ply'), 'ply')
        
        
        coords, face_index, colors = sample_surface(mesh, 10000, sample_color=True)
        
        rgb = colors[:,:3]
        rgb = rgb.astype(np.float32) / 255.0

        return coords, rgb


if __name__ == "__main__":
    import dataclasses
    guidance = lrm_Guidance(dataclasses.asdict(lrm_Guidance.Config()))
    coords, rgb = guidance("an astronaut riding a horse")
    print(coords.shape, rgb.shape)