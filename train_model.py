import torch
from torch import nn
from load_data import load_cn_vocab, load_en_vocab, load_train_data, get_batch_indices, load_test_data
from Transformer.transformer import Transformer

accuracy, loss, test_accuracy = None, None, None
BATCH_SIZE = 256
LR = 0.0001
D_MODEL = 512
D_FF = 2048
N_LAYERS = 6
HEADS = 8
DROPOUT_RATE = 0.2
N_EPOCHS = 500
PAD_ID = 0
MAX_LEN = 50

def main():
    global accuracy, loss, test_accuracy
    if torch.cuda.is_available():
        print("training on GPU")
    else:
        print("training on CPU")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cn2idx, idx2cn = load_cn_vocab()
    en2idx, idx2en = load_en_vocab()
    y, x = load_train_data()
    test_y, test_x = load_test_data()
    model = Transformer(len(en2idx), len(cn2idx), PAD_ID, D_MODEL, D_FF, N_LAYERS, HEADS, DROPOUT_RATE, MAX_LEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), LR)
    citerion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    for epoch in range(N_EPOCHS):
        for index, _ in get_batch_indices(len(x), BATCH_SIZE):
            x_batch = torch.LongTensor(x[index]).to(device)
            y_batch = torch.LongTensor(y[index]).to(device)
            y_input = y_batch[:, :-1]
            y_label = y_batch[:, 1:]
            y_hat = model(x_batch, y_input)

            y_label_mask = y_label != PAD_ID
            predict = torch.argmax(y_hat, -1)
            correct = predict == y_label
            accuracy = torch.sum(y_label_mask * correct) / torch.sum(y_label_mask)

            n, seq_len = y_label.shape
            y_hat = torch.reshape(y_hat, (n*seq_len, -1))
            y_label = torch.reshape(y_label, (n * seq_len,))
            loss = citerion(y_hat, y_label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

        for index, _ in get_batch_indices(len(test_x), BATCH_SIZE):
            x_batch_test = torch.LongTensor(test_x[index]).to(device)
            y_batch_test = torch.LongTensor(test_y[index]).to(device)
            y_input_test = y_batch_test[:, :-1]
            y_label_test = y_batch_test[:, 1:]
            y_hat_test = model(x_batch_test, y_input_test)

            y_label_mask_test = y_label_test != PAD_ID
            predict_test = torch.argmax(y_hat_test, -1)
            correct_test = predict_test == y_label_test
            test_accuracy = torch.sum(y_label_mask_test * correct_test) / torch.sum(y_label_mask_test)

        print(f"epoch {epoch}, loss {loss:4f}, accuracy: {accuracy:4f}, test accuracy: {test_accuracy:4f}")

    torch.save(model.state_dict(), "output/en_to_cn_model.pth")
    print("english to chinese model saved in output/en_to_cn_model.pth")

if __name__ == '__main__':
    main()