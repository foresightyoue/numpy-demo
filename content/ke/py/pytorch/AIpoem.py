# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 16:18:34 2020

@author: jy-3
"""

import numpy as np
    
datas = np.load("tang.npz", allow_pickle=True)
data = datas['data']
ix2word = datas['ix2word'].item()
word2ix = datas['word2ix'].item()

data[0]
print(type(word2ix))

data[0]

poem_txt = [ix2word[i] for i in data[0]]
poem_txt

import torch.nn as nn

class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        # 词向量层，词表大小 * 向量维度
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 网络主要结构
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=Config.num_layers)
        # 进行分类
        self.linear = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, input, hidden=None):
        seq_len, batch_size = input.size()
        #print(input.shape)
        if hidden is None:
            h_0 = input.data.new(Config.num_layers, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(Config.num_layers, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        # 输入 序列长度 * batch(每个汉字是一个数字下标)，
        # 输出 序列长度 * batch * 向量维度
        embeds = self.embeddings(input)
        # 输出hidden的大小： 序列长度 * batch * hidden_dim
        output, hidden = self.lstm(embeds, (h_0, c_0))
        output = self.linear(output.view(seq_len * batch_size, -1))
        return output, hidden
    
class Config(object):
    num_layers = 3  # LSTM层数
    data_path = 'data/'  # 诗歌的文本文件存放路径
    pickle_path = 'tang.npz'  # 预处理好的二进制文件
    author = None  # 只学习某位作者的诗歌
    constrain = None  # 长度限制
    category = 'poet.tang'  # 类别，唐诗还是宋诗歌(poet.song)
    lr = 1e-3
    weight_decay = 1e-4
    use_gpu = False
    epoch = 10
    batch_size = 16
    maxlen = 125  # 超过这个长度的之后字被丢弃，小于这个长度的在前面补空格
    plot_every = 200  # 每20个batch 可视化一次
    # use_env = True # 是否使用visodm
    env = 'poetry'  # visdom env
    max_gen_len = 200  # 生成诗歌最长长度
    debug_file = '/tmp/debugp'
    model_path = "tang_new.pth"  # 预训练模型路径
    prefix_words = '仙路尽头谁为峰？一见无始道成空。'  # 不是诗歌的组成部分，用来控制生成诗歌的意境
    start_words = '闲云潭影日悠悠'  # 诗歌开始
    acrostic = False  # 是否是藏头诗
    model_prefix = 'checkpoints/tang'  # 模型保存路径
    embedding_dim = 256
    hidden_dim = 512
    
import torch as t
import numpy as np
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from torchnet import meter
import tqdm

# 给定首句生成诗歌
def generate(model, start_words, ix2word, word2ix, prefix_words=None):
    results = list(start_words)
    start_words_len = len(start_words)
    # 第一个词语是<START>
    input = t.Tensor([word2ix['<START>']]).view(1, 1).long()
    if Config.use_gpu:
        input = input.cuda()
    hidden = None

    # 若有风格前缀，则先用风格前缀生成hidden
    if prefix_words:
        # 第一个input是<START>，后面就是prefix中的汉字
        # 第一个hidden是None，后面就是前面生成的hidden
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = input.data.new([word2ix[word]]).view(1, 1)

    # 开始真正生成诗句，如果没有使用风格前缀，则hidden = None，input = <START>
    # 否则，input就是风格前缀的最后一个词语，hidden也是生成出来的
    for i in range(Config.max_gen_len):
        output, hidden = model(input, hidden)
        # print(output.shape)
        # 如果还在诗句内部，输入就是诗句的字，不取出结果，只为了得到
        # 最后的hidden
        if i < start_words_len:
            w = results[i]
            input = input.data.new([word2ix[w]]).view(1, 1)
        # 否则将output作为下一个input进行
        else:
            # print(output.data[0].topk(1))
            top_index = output.data[0].topk(1)[1][0].item()
            w = ix2word[top_index]
            results.append(w)
            input = input.data.new([top_index]).view(1, 1)
        if w == '<EOP>':
            del results[-1]
            break
    return results


# 生成藏头诗
def gen_acrostic(model, start_words, ix2word, word2ix, prefix_words=None):
    result = []
    start_words_len = len(start_words)
    input = (t.Tensor([word2ix['<START>']]).view(1, 1).long())
    if Config.use_gpu:
        input = input.cuda()
    # 指示已经生成了几句藏头诗
    index = 0
    pre_word = '<START>'
    hidden = None

    # 存在风格前缀，则生成hidden
    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = (input.data.new([word2ix[word]])).view(1, 1)

    # 开始生成诗句
    for i in range(Config.max_gen_len):
        output, hidden = model(input, hidden)
        top_index = output.data[0].topk(1)[1][0].item()
        w = ix2word[top_index]
        # 说明上个字是句末
        if pre_word in {'。', '，', '?', '！', '<START>'}:
            if index == start_words_len:
                break
            else:
                w = start_words[index]
                index += 1
                # print(w,word2ix[w])
                input = (input.data.new([word2ix[w]])).view(1, 1)
        else:
            input = (input.data.new([top_index])).view(1, 1)
        result.append(w)
        pre_word = w
    return result


import torch as t
import numpy as np
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from torchnet import meter
import tqdm


def train():
    if Config.use_gpu:
        Config.device = t.device("cuda")
    else:
        Config.device = t.device("cpu")
    device = Config.device
    # 获取数据
    datas = np.load("tang.npz", allow_pickle=True)
    data = datas['data']
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()
    data = t.from_numpy(data)
    dataloader = DataLoader(data,
                            batch_size=Config.batch_size,
                            shuffle=True,
                            num_workers=2)

    # 定义模型
    model = PoetryModel(len(word2ix),
                        embedding_dim=Config.embedding_dim,
                        hidden_dim = Config.hidden_dim)
    Configimizer = optim.Adam(model.parameters(),lr=Config.lr)
    criterion = nn.CrossEntropyLoss()
    if Config.model_path:
        model.load_state_dict(t.load(Config.model_path,map_location='cpu'))
    # 转移到相应计算设备上
    model.to(device)
    loss_meter = meter.AverageValueMeter()
    # 进行训练
    f = open('result.txt','w')
    for epoch in range(Config.epoch):
        loss_meter.reset()
        for li,data_ in tqdm.tqdm(enumerate(dataloader)):
            #print(data_.shape)
            data_ = data_.long().transpose(1,0).contiguous()
            # 注意这里，也转移到了计算设备上
            data_ = data_.to(device)
            Configimizer.zero_grad()
            # n个句子，前n-1句作为输入，后n-1句作为输出，二者一一对应
            input_,target = data_[:-1,:],data_[1:,:]
            output,_ = model(input_)
            #print("Here",output.shape)
            
            print(target.shape,target.view(-1).shape)
            loss = criterion(output,target.view(-1))
            loss.backward()
            Configimizer.step()
            loss_meter.add(loss.item())
            # 进行可视化
            if (1+li)%Config.plot_every == 0:
                print("训练损失为%s"%(str(loss_meter.mean)))
                f.write("训练损失为%s"%(str(loss_meter.mean)))
                for word in list(u"春江花朝秋月夜"):
                    gen_poetry = ''.join(generate(model,word,ix2word,word2ix))
                    print(gen_poetry)
                    f.write(gen_poetry)
                    f.write("\n\n\n")
                    f.flush()
        t.save(model.state_dict(),'%s_%s.pth'%(Config.model_prefix,epoch))


if __name__ == '__main__':
    train()
    

import torch as t



def userTest():
    print("正在初始化......")
    datas = np.load("tang.npz", allow_pickle=True)
    data = datas['data']
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()
    model = PoetryModel(len(ix2word), Config.embedding_dim, Config.hidden_dim)
    model.load_state_dict(t.load(Config.model_path, 'cpu'))
    if Config.use_gpu:
        model.to(t.device('cuda'))
    print("初始化完成！\n")
    while True:
        print("欢迎使用唐诗生成器，\n"
              "输入1 进入首句生成模式\n"
              "输入2 进入藏头诗生成模式\n")
        mode = int(input())
        if mode == 1:
            print("请输入您想要的诗歌首句，可以是五言或七言")
            start_words = str(input())
            gen_poetry = ''.join(generate(model, start_words, ix2word, word2ix))
            print("生成的诗句如下：%s\n" % (gen_poetry))
        elif mode == 2:
            print("请输入您想要的诗歌藏头部分，不超过16个字，最好是偶数")
            start_words = str(input())
            gen_poetry = ''.join(gen_acrostic(model, start_words, ix2word, word2ix))
            print("生成的诗句如下：%s\n" % (gen_poetry))
            #print("生成的诗句如下：%s\n" % ("浩歌夜坐生光塘，然余坏竹入袁墙。最爱林泉多往事，喜逢日月共流光。欢讴未暇听雷响，芷壑已惊蛛雁忙。若无一年离世曰，宝莲山中有仙郎。"))


if __name__ == '__main__':
    userTest()