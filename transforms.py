from torchvision.transforms import transforms

# Cityscapes
train_transform = transforms.Compose(
    [
        transforms.Resize((512, 1024)),
        transforms.ToTensor(),
    ]
)
label_transform = transforms.Resize((512, 1024))

eval_transform = transforms.Compose([transforms.ToTensor()])

# GTA5
gta_train_transform = transforms.Compose(
    [
        transforms.Resize((512, 1024), transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ]
)
gta_label_transform = transforms.Resize(
    (512, 1024), transforms.InterpolationMode.NEAREST
)
gta_val_transform = transforms.Compose([transforms.ToTensor()])
