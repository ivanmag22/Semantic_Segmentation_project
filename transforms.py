from torchvision.transforms import transforms

# Cityscapes
# transforms.CenterCrop((512,1024))
train_transform = transforms.Compose(
    [
        transforms.Resize(
            (512, 1024), transforms.InterpolationMode.BILINEAR
        ),  # BILINEAR because we are dealing with continuous values
        transforms.ToTensor(),
    ]
)
label_transform = transforms.Resize(
    (512, 1024), transforms.InterpolationMode.NEAREST
)  # NEAREST because we are dealing with discrete values [0,18]
eval_transform = transforms.Compose([transforms.ToTensor()])

# GTA5
gta_train_transform = transforms.Compose(
    [
        transforms.Resize(
            (512, 1024), transforms.InterpolationMode.BILINEAR
        ),  # BILINEAR because we are dealing with continuous values
        transforms.ToTensor(),
    ]
)
gta_label_transform = transforms.Resize(
    (512, 1024),
    transforms.InterpolationMode.NEAREST,  # NEAREST because we are dealing with discrete values [0,18]
)
gta_val_transform = transforms.Compose([transforms.ToTensor()])
