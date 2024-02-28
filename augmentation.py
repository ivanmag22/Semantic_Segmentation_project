from torchvision.transforms import transforms

# data augmentation
bright_t = transforms.ColorJitter(brightness=[1, 2])
contrast_t = transforms.ColorJitter(contrast=[2, 5])
saturation_t = transforms.ColorJitter(saturation=[1, 3])
hue_t = transforms.ColorJitter(hue=0.2)
gs_t = transforms.Grayscale(3)
hflip_t = transforms.RandomHorizontalFlip(p=1)
rp_t = transforms.RandomPerspective(p=1, distortion_scale=0.5)

augmentation_transforms = [
    bright_t,
    contrast_t,
    saturation_t,
    hue_t,
    gs_t,
    hflip_t,
    rp_t,
]

"""
bright_t = transforms.ColorJitter(brightness=[1,2])
contrast_t = transforms.ColorJitter(contrast = [2,5])
saturation_t = transforms.ColorJitter(saturation = [1,3])
hue_t = transforms.ColorJitter(hue = 0.2)
gs_t = transforms.Grayscale(3)
hflip_t = transforms.RandomHorizontalFlip(p = 1)
rp_t = transforms.RandomPerspective(p = 1, distortion_scale = 0.5)
rot_t = transforms.RandomRotation(degrees = 90)

aug_transformations = {
    "CS-HF": transforms.Compose([contrast_t, saturation_t, hflip_t]),
    "H-RP": transforms.Compose([hue_t, rp_t]),
    "B-GS-R": transforms.Compose([bright_t, gs_t, rot_t])
    }
"""
