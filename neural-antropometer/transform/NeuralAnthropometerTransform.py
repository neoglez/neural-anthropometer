import torch
import numpy as np
from torchvision import transforms

class TwoDToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        
        pil2tensor = transforms.ToTensor()
        #tensor2pil = transforms.ToPILImage()
        #sample["image"] = image
        #sample["imagefile"] = imagefile
        hbd = []
        # Use this parameter to control the tensor dtype.
        tensor_dtype = torch.float32
        if "image" in sample:
            image = sample["image"]
            sample["image"] = pil2tensor(image)
        if (type(sample['annotations']['human_dimensions'])
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


if __name__ == "__main__":
    simulated_sample = {
        "image": np.random.randint(low=0, high=254, size=(5, 7), dtype=np.int),
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
