import torch
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import numpy as np


def modify_vocab(vocab):
    embedding_dim = vocab.vector_size
    weights = vocab.vectors
    pad_vector = np.zeros(embedding_dim)  
    unk_vector = np.mean(weights, axis=0)   
    if '<pad>' not in vocab:
        vocab.add_vector('<pad>', pad_vector)

    if '<unk>' not in vocab:
        vocab.add_vector('<unk>', unk_vector)
    return vocab


def train(model, data_loader, validation_loader, test_loader, config, criterion, optimizer, device):
    print(f"Using device: {device}")
    loss_dict = {
        'train_loss': [],
        'validation_loss': [],
        'test_loss': []
    }
    accuracy_dict = {
        'validation_acc': [],
        'test_acc': []
    }
    best_loss = float('inf')
    patience = 5  # 耐心值
    counter = 0
    epoch_counter = 0
    lambda_l1 = 0.001
    for epoch in range(config['num_epochs']):
        model.train()
        avg_loss = 0
        total_loss = 0
        for i, (inputs, labels) in enumerate(data_loader):
            input_text, input_aspect, labels = inputs[0].to(device), inputs[1].to(device), labels.to(device) 
            outputs, _, _ = model(input_text, input_aspect)
            loss = criterion(outputs, labels)

            # L1正则化
            l1_penalty = 0
            for param in model.parameters():
                l1_penalty += torch.abs(param).sum()

            loss += l1_penalty * lambda_l1
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{config["num_epochs"]}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item():.4f}')
        avg_loss = total_loss / len(data_loader)
        validation_dict = evaluate(model, validation_loader, criterion, device)   
        test_dict = evaluate(model, test_loader, criterion, device)
        loss_dict['train_loss'].append(avg_loss)
        loss_dict['validation_loss'].append(validation_dict["avg_loss"])
        loss_dict['test_loss'].append(test_dict["avg_loss"])
        accuracy_dict['validation_acc'].append(validation_dict['accuracy'])
        accuracy_dict['test_acc'].append(test_dict['accuracy'])
        print(f'Epoch [{epoch+1}/{config["num_epochs"]}]:')
        print(f'train_avg_loss: {avg_loss:.4f}')
        print(f'validation_avg_loss: {validation_dict["avg_loss"]:.4f}, validation_acc: {validation_dict["accuracy"]:.4f}')
        print(f'test_avg_loss: {test_dict["avg_loss"]:.4f}, test_acc: {test_dict["accuracy"]:.4f}')

        if validation_dict["avg_loss"] < best_loss:
            best_loss = validation_dict["avg_loss"]
            counter = 0
            torch.save(model.state_dict(), './trained_model/best_model.pth')  # 保存最佳模型
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping after {epoch+1} epochs.")
                epoch_counter = epoch + 1
                break
    # 保存
    # torch.save(model.state_dict(), 'cnn_gate_aspect_300_test2.ckpt')
    return loss_dict, accuracy_dict, epoch_counter

def evaluate(model, data_loader, criterion, device):
    model.eval()
    true_labels = []
    predictions = []
    total_loss = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            input_text, input_aspect, labels = inputs[0].to(device), inputs[1].to(device), labels.to(device)
            outputs, _, _ = model(input_text, input_aspect)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            true_labels.extend(labels.tolist())
            predictions.extend(predicted.tolist())


    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')
    f1 = f1_score(true_labels, predictions, average='macro')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'avg_loss': avg_loss
    }

