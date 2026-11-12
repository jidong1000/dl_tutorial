import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import re

pwd = r"D:\project\deep_learning\dl_tutorial\data\poems.txt"

# 数据预处理
def preprocess_data():
    poems = []
    data_set = set()
    with open(pwd, 'r', encoding='utf-8') as f:
        for line in f:
            # 去掉标点和两侧空白
            # strip():去除两侧空白
            line = re.sub(r"[，。？！、：]", "", line).strip()
            # update():将可遍历对象的每个元素都放入集合中，并且去重
            data_set.update(list(line))
            poems.append(list(line))
    # 构建词表
    id2word = list(data_set) + ['<UNK>']
    word2id = {word:id for id, word in enumerate(id2word)}
    # 将诗句id化
    id_seqs = []
    for poem in poems:
        id_seq = [word2id.get(word) for word in poem]
        id_seqs.append(id_seq)

    return id_seqs, id2word, word2id

id_seqs, id2word, word2id = preprocess_data()

# 构建数据集
class PoetryDataset(Dataset):
    def __init__(self, id_seqs, seq_len):
        self.seq_len = seq_len
        self.data = []
        for id_seq in id_seqs:
            for i in range(len(id_seq) - self.seq_len):
                self.data.append((id_seq[i:i + self.seq_len], id_seq[i + 1:i + 1 + self.seq_len])) 

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.LongTensor(self.data[idx][0])
        y = torch.LongTensor(self.data[idx][1])

        return x, y

dataset = PoetryDataset(id_seqs, 24)

# 创建RNN模型
class PoetryRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1):
        super().__init__()
        # 词嵌入层
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        # RNN层
        self.RNN = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True)
        # 全连接层
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, hx=None):
        embed = self.embed(input)
        output, hn = self.RNN(embed, hx)
        output = self.linear(output)

        return output, hn

model = PoetryRNN(vocab_size=len(id2word), embedding_dim=256, hidden_size=512, num_layers=2)

# 模型训练
def train(model, dataset, lr, epoch_num, batch_size, device):
    model.to(device)
    model.train()
    # 定义损失函数和优化器
    # 多分类任务的损失函数一般选择交叉熵
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练
    for epoch in range(epoch_num):
        train_loss = 0
        dataloader = DataLoader(dataset, batch_size, shuffle=True)
        for batch_index, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            # 前向传播
            y_pred, _ = model(x)
            # 计算损失
            loss_value = loss(y_pred.transpose(1, 2), y)
            # 反向传播
            loss_value.backward()
            # 更新参数
            optimizer.step()
            # 梯度清零
            optimizer.zero_grad()

            train_loss += loss_value.item() * x.shape[0]
            print(f"\repoch:{epoch + 1:0>2}[{'=' * int((batch_index + 1) / len(dataloader) * 50)}]", end='')

        # 计算平均损失
        this_loss = train_loss / len(dataset)
        print(f"train loss:{this_loss:.4f}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.001
epoch_num = 20
batch_size = 32

train(model, dataset, lr, epoch_num, batch_size, device)

# 生成新的
def generate_poem(model, id2word, word2id, init_token, line_num=4, col_num=7):
    model.eval()
    # 记录生成结果
    poem = []
    rest_col_num = col_num
    init_id = word2id.get(init_token, word2id['<UNK>'])
    if init_id != word2id['<UNK>']:
        poem.append(init_token)
        rest_col_num -= 1
    input = torch.LongTensor([[init_id]]).to(device)

    with torch.no_grad():
        for _ in range(line_num):
            for interpunction in ['，', '。\n']:
                while(rest_col_num > 0):
                    output, _ = model(input)
                    # output的维度是1 * 1 * vocab_size
                    prob = torch.softmax(output[0, 0], dim=-1)
                    next_id = torch.multinomial(prob, num_samples=1)
                    poem.append(id2word[next_id.item()])
                    # 更新循环
                    rest_col_num -= 1
                    # next_id是一个向量，需要与[[init_id]]保持一致
                    input = next_id.unsqueeze(0)
                poem.append(interpunction)
                rest_col_num = col_num

    return ''.join(poem)

print(generate_poem(model, id2word, word2id, '一', line_num=4, col_num=7))
