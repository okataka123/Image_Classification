import torchvision.transforms as transforms

def get_train_transformation(augmentation=False):
    '''
    学習時の変換（データ拡張の有無で異なる）
    '''
    if augmentation:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224), # ランダムにリサイズとクロップ
            transforms.RandomHorizontalFlip(), # ランダムに水平反転
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), # カラー調整
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 正規化(ImageNetの場合)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),      # リサイズ
            transforms.CenterCrop(224),  # 中心をクロップ
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 正規化(ImageNetの場合)
        ])
    return transform


def get_test_transformation():
    '''
    テスト時の変換（データ拡張は通常行わない）
    '''
    transform = transforms.Compose([
        transforms.Resize(256),      # リサイズ
        transforms.CenterCrop(224),  # 中心をクロップ
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 正規化(ImageNetの場合)
    ])
    return transform
