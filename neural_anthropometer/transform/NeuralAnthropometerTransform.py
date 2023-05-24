import torch
import numpy as np
from torchvision import transforms
from PIL import Image

class TwoDToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        
        pil2tensor = transforms.ToTensor()
        hbd = []
        # Use this parameter to control the tensor dtype.
        tensor_dtype = torch.float32
        if 'image' in sample:
            image = sample["image"]
            sample['image'] = pil2tensor(image)
        if ('annotations' in sample and
            'human_dimensions' in sample['annotations'] and
            type(sample['annotations']['human_dimensions'])
            is not torch.Tensor):
            
            for i, e in enumerate(
                    ['chest_circumference', 'height', 'inseam',
                     'left_arm_length', 'pelvis_circumference', 
                     'right_arm_length', 'shoulder_width',
                     'waist_circumference']):
                
                if e in sample['annotations']['human_dimensions']:
                    hbd.insert(i, sample['annotations']['human_dimensions'][e])
            
            sample['annotations']['human_dimensions'] = torch.tensor(hbd,
                                                     dtype=tensor_dtype)

    # The equivalent tensor contains the information in the corresponding
    # integer indices.
        # # print(inputs.shape)

        # dimensions = np.array(
        #     [
        #         sample["annotations"]["human_dimensions"][human_dim]
        #         for human_dim in sample["annotations"]["human_dimensions"]
        #     ]
        # )

        
        # return {
        #     "image": pil2tensor(),
        #     "annotations": {
        #         "human_dimensions": torch.tensor(
        #             [dimensions], dtype=tensor_dtype
        #         )
        #     },
        #     "imagefile": sample["imagefile"],
        #     "annotation_file": sample["annotation_file"],
        # }
        
        return sample

class SquareAndGreyscaleImage(object):
    """Transform an image such as it can be feed to the neural
        anthropometer. Usually that implies escaling the image
        keeping its aspect ratio and converting the image to
        greyscale.
        Code heavily inspired on
        https://note.nkmk.me/en/python-pillow-add-margin-expand-canvas/
    """
    
    def __init__(self, shape=(200,200), background_color=(0,0,0)):
        self.shape = shape
        self.background_color = background_color

    def __call__(self, sample):
        wasTensor = False
        if "image" in sample:
            image = sample["image"]
            
            # If the image is not PIL, we have to conver it.
            if (torch.is_tensor(image)):
                toPIL = transforms.ToPILImage()
                image = toPIL(image)
                wasTensor = True
            # At this point we assume it is a PIL image.
            
            # 1.- Scale
            image.thumbnail(self.shape, Image.Resampling.LANCZOS)
            
            # 2.- Reshape to square size                
            width, height = image.size
            # if the image is already square is OK
            if width != height:
                if width > height:
                    result = Image.new(image.mode, (width, width),
                                       self.background_color)
                    result.paste(image, (0, (width - height) // 2))
                else:
                    result = Image.new(image.mode, (height, height),
                                       self.background_color)
                    result.paste(image, ((height - width) // 2, 0))
            
            # 3.- Convert to greyscale
            result = result.convert("L")
            
            pil2tensor = transforms.ToTensor()
            sample["image"] = pil2tensor(result) if wasTensor else result          
        
        return sample


if __name__ == "__main__":
    simulated_sample = {
        "image": np.random.randint(low=0, high=254, size=(500, 700), dtype=np.int16),
        "annotations": {"human_dimensions": {"dim1": 1, "dim2": 2, "dim3": 3}},
        "imagefile": "simulated_image_filename.png",
        "annotation_file": "simulated_annotation_filename.json",
    }
    transform = TwoDToTensor()

    print(type(simulated_sample))
    print(type(simulated_sample["image"]))
    print(type(simulated_sample["annotations"]["human_dimensions"]))
    print(simulated_sample["annotations"]["human_dimensions"])

    print(simulated_sample["image"].shape)

    trasformed_sample = transform(simulated_sample)

    print(type(simulated_sample))
    print(type(trasformed_sample["image"]))
    print(type(trasformed_sample["annotations"]["human_dimensions"]))
    print(trasformed_sample["annotations"]["human_dimensions"])

    print(trasformed_sample["image"].shape)
    print(trasformed_sample["annotations"]["human_dimensions"].shape)
    
    # RGB image
    simulated_sample = {
        "image": np.random.randint(low=0, high=128, size=(500, 700, 3), dtype=np.uint8)
    }
    transform = transforms.Compose([TwoDToTensor(), SquareAndGreyscaleImage()])
    
    trasformed_sample = transform(simulated_sample)

    print(type(simulated_sample))
    print(type(trasformed_sample["image"]))
    
    # greyscale image
    simulated_sample = {
        "image": np.random.randint(low=0, high=255, size=(5, 7), dtype=np.int16)
    }
    transform = transforms.Compose([
        TwoDToTensor(),
        # one background color
        SquareAndGreyscaleImage(background_color=(0))])
    
    trasformed_sample = transform(simulated_sample)

    print(type(simulated_sample))
    print(type(trasformed_sample["image"]))
    
