from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, df, transform):
        """
        Initializes the CustomDataset dataset.

        :param df: The input dataframe containing image paths and labels.
        :type df: pandas.DataFrame
        :param transform: The transformation to apply to the images.
        :type transform: torchvision.transforms.Transform
        """
        self.df = df
        self.transform = transform
        self.image_paths = self.df.image_path.values.tolist()
        self.labels = self.df.labels_encoded.values.tolist()

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        :return: The length of the dataset.
        :rtype: int
        """
        return len(self.df)

    def __getitem__(self, index):
        """
        Retrieves a specific sample from the dataset.

        :param index: The index of the sample to retrieve.
        :type index: int
        :return: The image and label of the sample.
        :rtype: torch.Tensor, int
        """
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image, label, image_path
