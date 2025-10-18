import torch
from Transformer.transformer import Transformer
from load_data import load_cn_vocab, load_en_vocab, idx_to_sentence

BATCH_SIZE = 1
LR = 0.0001
D_MODEL = 512
D_FF = 2048
N_LAYERS = 6
HEADS = 8
DROPOUT_RATE = 0.2
N_EPOCHS = 150
PAD_ID = 0
MAX_LEN = 50

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cn2idx, idx2cn = load_cn_vocab()
    en2idx, idx2en = load_en_vocab()

    model = Transformer(len(en2idx), len(cn2idx), PAD_ID, D_MODEL, D_FF, N_LAYERS, HEADS, DROPOUT_RATE, MAX_LEN)
    model.to(device)
    model.eval()

    model_path = 'output/en_to_cn_model.pth'
    model.load_state_dict(torch.load(model_path))

    my_input = ['we', 'should', 'protect', 'the', 'environment']
    x_batch = torch.LongTensor([[en2idx[x] for x in my_input]]).to(device)

    cn_sentence = idx_to_sentence(x_batch[0], idx2en, True)
    print(cn_sentence)

    y_input = torch.ones(BATCH_SIZE, MAX_LEN, dtype=torch.long).to(device) * PAD_ID
    y_input[0] = en2idx['<S>']

    with torch.no_grad():
        for i in range(1, y_input.shape[1]):
            y_hat = model(x_batch, y_input)
            for j in range(BATCH_SIZE):
                y_input[j, i] = torch.argmax(y_hat[j, i - 1])
                print(idx_to_sentence(y_input[0], idx2cn, True))
    output_sentence = idx_to_sentence(y_input[0], idx2cn, True)
    print("\n")
    print("translated sentence:" + output_sentence)

if __name__ == '__main__':
    main()