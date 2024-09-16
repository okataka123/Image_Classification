import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm
from models.cnn import SimpleCNN
from data.dataset_loader import get_data_loader


def train(config, save_model=False):
    model = SimpleCNN()
    #optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = get_data_loader(config['dataset'], batch_size=config['batch_size'], train=True)
    test_loader = get_data_loader(config['dataset'], batch_size=config['batch_size'], train=False)
    
    train_loss_value = []
    train_acc_value = []
    test_loss_value = []
    test_acc_value = []

    epochs = config['epochs']

    for _ in tqdm(range(epochs)):
        sum_loss = 0
        sum_acc = 0
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            predicts = torch.argmax(outputs, dim=1)
            acc = torch.sum(predicts == labels)/len(labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            sum_acc += acc.item()
        #train_loss_value.append(sum_loss)
        train_loss_value.append(sum_loss/len(train_loader))
        train_acc_value.append(sum_acc/len(train_loader))

        # 学習したモデルを用いて、テストデータで推論する。
        sum_loss = 0
        sum_acc = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                predicts = torch.argmax(outputs, dim=1)
                acc = torch.sum(predicts == labels)/len(labels)
                loss = criterion(outputs, labels)
                sum_loss += loss.item()
                sum_acc += acc.item()
        test_loss_value.append(sum_loss/len(test_loader))
        test_acc_value.append(sum_acc/len(test_loader))

    # 全テストデータを使って推論（confusion matrix用）
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicts = torch.argmax(outputs, dim=1)
            all_preds += predicts
            all_labels += labels

    visualization_loss(train_loss_value, test_loss_value)
    visualization_acc(train_acc_value, test_acc_value)
    visualization_confusion_matrix(all_labels, all_preds)

    if save_model:
        torch.save()
        pass


def test():
    '''
    学習済みモデルを使って推論。
    '''
    test_loader = get_data_loader('MNIST', batch_size=100, train=False)
    test_loss_value = []
    test_acc_value = []
    pass


def visualization_loss(train_loss_value, test_loss_value):
    '''
    epochごとのtrain lossとtest lossの推移グラフを可視化
    '''
    plt.figure(figsize=(6,6))
    plt.plot(range(len(train_loss_value)), train_loss_value, label='train loss')
    plt.plot(range(len(test_loss_value)), test_loss_value, label='test loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


def visualization_acc(train_acc_value, test_acc_value):
    '''
    epochごとのaccuracyの推移グラフを可視化
    '''
    plt.figure(figsize=(6,6))
    plt.plot(range(len(train_acc_value)), train_acc_value, label='train acc')
    plt.plot(range(len(test_acc_value)), test_acc_value, label='test acc')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


def visualization_confusion_matrix(all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()    

if __name__ == '__main__':
    import yaml
    with open('configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print('config =', config)
    train(config)



