import os
import torch

from .common import ImageFolderWithPaths, SubsetSampler
from .imagenet_classnames import get_classnames
import numpy as np
from src.models.augmentations import *
from torchvision import transforms


mixture_width = 3
mixture_depth=1
all_ops=False
no_jsd=False
# def aug(image, preprocess):
#     """Perform AugMix augmentations and compute mixture.

#     Args:
#         image: PIL.Image input image
#         preprocess: Preprocessing function which should return a torch tensor.

#     Returns:
#         mixed: Augmented and mixed image.
#     """
#     aug_list = augmentations.augmentations
#     if all_ops:
#         aug_list = augmentations.augmentations_all

#     ws = np.float32(np.random.dirichlet([1] * mixture_width))
#     m = np.float32(np.random.beta(1, 1))

#     mix = torch.zeros_like(preprocess(image))
#     for i in range(mixture_width):
#         image_aug = image.copy()
#         depth = mixture_depth if mixture_depth > 0 else np.random.randint(
#             1, 4)
#         for _ in range(depth):
#             op = np.random.choice(aug_list)
#             image_aug = op(image_aug, args.aug_severity)
#             # Preprocessing commutes since all coefficients are convex
#         mix += ws[i] * preprocess(image_aug)

#     mixed = (1 - m) * preprocess(image) + m * mix
#     return mixed

# class AugMixDataset(torch.utils.data.Dataset):
#     """Dataset wrapper to perform AugMix augmentation."""

#     def __init__(self, dataset, preprocess, no_jsd=False):
#         self.dataset = dataset
#         self.preprocess = preprocess
#         self.no_jsd = no_jsd

#     def __getitem__(self, i):
#         print(self.dataset[i])
#         print(len(self.dataset[i]))
#         x, y = self.dataset[i]
#         if self.no_jsd:
#             return aug(x, self.preprocess), y
#         else:
#             im_tuple = (self.preprocess(x), aug(x, self.preprocess),
#                         aug(x, self.preprocess))
#             return im_tuple, y

#     def __len__(self):
#         return len(self.dataset)


class ImageNet:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=32,
                 classnames='openai'):
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classnames = get_classnames(classnames)

        self.populate_train()
        self.populate_test()
    
    def populate_train(self):
        traindir = os.path.join(self.location, self.name(), 'train')
        # preprocess = transforms.Compose(
        #     [transforms.ToTensor(),
        #     transforms.Normalize([0.5] * 3, [0.5] * 3)])
        preprocess = transforms.Compose(
            [transforms.ToTensor()])
        self.train_dataset = ImageFolderWithPaths(
            traindir,
            transform=self.preprocess,preprocess=preprocess,augmix=True)
        
        sampler = self.get_train_sampler()
        kwargs = {'shuffle' : True} if sampler is None else {}
        # self.train_dataset = AugMixDataset(self.train_dataset, preprocess, no_jsd)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            **kwargs,
        )

    def populate_test(self):
        self.test_dataset = self.get_test_dataset()
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.get_test_sampler()
        )

    def get_test_path(self):
        test_path = os.path.join(self.location, self.name(), 'val_in_folder')
        if not os.path.exists(test_path):
            test_path = os.path.join(self.location, self.name(), 'val')
        return test_path

    def get_train_sampler(self):
        return None

    def get_test_sampler(self):
        return None

    def get_test_dataset(self):
        return ImageFolderWithPaths(self.get_test_path(), transform=self.preprocess)

    def name(self):
        return 'imagenet'

class ImageNetTrain(ImageNet):

    def get_test_dataset(self):
        pass

class ImageNetK(ImageNet):

    def get_train_sampler(self):
        idxs = np.zeros(len(self.train_dataset.targets))
        target_array = np.array(self.train_dataset.targets)
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:self.k()] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetSampler(np.where(idxs)[0])
        return sampler


def project_logits(logits, class_sublist_mask, device):
    if isinstance(logits, list):
        return [project_logits(l, class_sublist_mask, device) for l in logits]
    if logits.size(1) > sum(class_sublist_mask):
        return logits[:, class_sublist_mask].to(device)
    else:
        return logits.to(device)

class ImageNetSubsample(ImageNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        class_sublist, self.class_sublist_mask = self.get_class_sublist_and_mask()
        self.classnames = [self.classnames[i] for i in class_sublist]

    def get_class_sublist_and_mask(self):
        raise NotImplementedError()

    def populate_train(self):
        pass

    def project_logits(self, logits, device):
        return project_logits(logits, self.class_sublist_mask, device)

class ImageNetSubsampleValClasses(ImageNet):
    def get_class_sublist_and_mask(self):
        raise NotImplementedError()

    def populate_train(self):
        pass
    
    def get_test_sampler(self):
        self.class_sublist, self.class_sublist_mask = self.get_class_sublist_and_mask()
        idx_subsample_list = [range(x * 50, (x + 1) * 50) for x in self.class_sublist]
        idx_subsample_list = sorted([item for sublist in idx_subsample_list for item in sublist])
        
        sampler = SubsetSampler(idx_subsample_list)
        return sampler

    def project_labels(self, labels, device):
        projected_labels = [self.class_sublist.index(int(label)) for label in labels]
        return torch.LongTensor(projected_labels).to(device)

    def project_logits(self, logits, device):
        return project_logits(logits, self.class_sublist_mask, device)

ks = [1, 2, 4, 8, 16, 25, 32, 50, 64, 128, 600]

for k in ks:
    cls_name = f"ImageNet{k}"
    dyn_cls = type(cls_name, (ImageNetK, ), {
        "k": lambda self, num_samples=k: num_samples,
    })
    globals()[cls_name] = dyn_cls