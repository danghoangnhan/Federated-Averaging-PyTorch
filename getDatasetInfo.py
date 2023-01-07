import torch
from torchvision import datasets, transforms

IMAGE_SIZE = 28

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)
train_dataset = datasets.MNIST(root="../Experiment/data/MNIST", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="../Experiment/data/MNIST", train=False, download=True, transform=transform)

# train_dataset[0][0].numpy()
print(len(train_dataset))
# a[0][0][0]=1
# print(len(a[0][0]))
# print(a[0][0])

# plt.imshow(a[0])
# plt.show()

# tmp_img=train_dataset[0][0]
# tmp_label=train_dataset[0][1]
num_of_class_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# print(type(tmp_img))
# print(tmp_img[0])
# print(tmp_label)
# plt.imshow(train_dataset[1][0].numpy())
# print(train_dataset)

# number of each class
for i in range(60000):
    index = train_dataset[i][1]
    num_of_class_list[index] = num_of_class_list[index] + 1
print(num_of_class_list)
# print(type(a))


# for i in range(2):
# a=train_dataset[i][0].numpy().reshape(28,28)
# print(train_dataset[i][1])
# plt.imshow(a)
# plt.show()
# print(train_dataset[1].label)
