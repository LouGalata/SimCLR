from data_aug.load_laboratory_dataset import LaboratoryDataset
from torchvision import transforms

if __name__ == "__main__":
    root_folder = 'datasets'
    transform = transforms.Compose([transforms.ToTensor()])
    labdata = LaboratoryDataset(root_folder, split='unlabeled', transform=transform)
    pass
