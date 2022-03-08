import os
import sys
import numpy as np
import tensorflow as tf
import json
import pickle
import collections
import Vocabulary
import Config
import utils


class BatchInput(collections.namedtuple("BatchInput",
                                        ("key_input", "val_input", "input_lens",
                                         "target_input", "target_output", "output_lens",
                                         "group", "group_lens", "group_cnt",
                                         "target_type", "target_type_lens",
                                         "text", "slens",
                                         "category"))):
    pass

class EPWDataset:
    def __init__(self):
        self.config = Config.config
        if not os.path.exists(self.config.vocab_file):
            pickle.dump(Vocabulary.Vocabulary(), open(self.config.vocab_file, "wb"))
        self.vocab = pickle.load(open(self.config.vocab_file, "rb"))
        utils.print_out("finish reading vocab : {}".format(len(self.vocab.id2word)))
        self.cate2FK = {
            "裙": ["类型", "版型", "材质", "颜色", "风格", "图案", "裙型", "裙下摆", "裙腰型", "裙长", "裙衣长", "裙袖长", "裙领型", "裙袖型", "裙衣门襟",
                  "裙款式"],
            "裤": ["类型", "版型", "材质", "颜色", "风格", "图案", "裤长", "裤型", "裤款式", "裤腰型", "裤口"],
            "上衣": ["类型", "版型", "材质", "颜色", "风格", "图案", "衣样式", "衣领型", "衣长", "衣袖长", "衣袖型", "衣门襟", "衣款式"]}
        for key, val in self.cate2FK.items():
            self.cate2FK[key] = dict(zip(val, range(len(val)))) # 3元组 (裙: 类型：1, 版型：2)

        self.input_graph = tf.Graph()
        with self.input_graph.as_default():
            proto = tf.ConfigProto()
            proto.gpu_options.allow_growth = True
            self.input_sess = tf.Session(config=proto)
            self.prepare_dataset()

    def get_batch(self, data):
        with self.input_graph.as_default():
            input_initializer, batch = self.make_iterator(data)
            self.input_sess.run(input_initializer)
        return batch

    def next_batch(self, batch):
        with self.input_graph.as_default():
            res = self.input_sess.run(batch)
        return res

    def prepare_dataset(self):
        with self.input_graph.as_default():
            self.dev = self.get_dataset(self.config.dev_file, False)
            self.test = self.get_dataset(self.config.test_file, False)
            self.train = self.get_dataset(self.config.train_file, True)
    
    def make_iterator(self, data):
        iterator = data.make_initializable_iterator()
        (key_input, val_input, input_lens,
         target_input, target_output, output_lens,
         group, group_lens, group_cnt,
         target_type, target_type_lens,
         text, slens,
         category) = iterator.get_next()
        return iterator.initializer, \
               BatchInput(
                   key_input=key_input,
                   val_input=val_input,
                   input_lens=input_lens,

                   target_input=target_input,
                   target_output=target_output,
                   output_lens=output_lens,

                   group=group,
                   group_lens=group_lens,
                   group_cnt=group_cnt,

                   target_type=target_type,
                   target_type_lens=target_type_lens,

                   text=text,
                   slens=slens,

                   category=category
               )

    def sort(self, cate, lst):
        assert cate in self.cate2FK #如果不在大类里面，抛出异常
        # tgt是 初始化的字典里后面包含的字典
        tgt = self.cate2FK[cate]
        # 此时lst中的每个值为x, 返回对应x在tgt中的值，无的话返回len(tgt)+1
        return sorted(lst, key=lambda x: tgt.get(x[0], len(tgt) + 1))

    def process_inst(self, line):
        res = {"feats" + suffix: [] for suffix in ['_key', '_val']}
        # line是dic 是输入数据的一行  line['feature']是里面的属性  类型是其中一个属性，可以看做k-v，这里目的是把这个k-v提出来作为一个大类
        # cate 字符串
        cate = dict(line['feature'])['类型']
        val_tpe = 1
        # 对应类，按照指定类下的字典k-v进行排序
        feats = self.sort(cate, line['feature'])
        # 拆到res中，把k-v的字符串转换成词表中的id
        for item in feats:
            res["feats_key"].append(self.vocab.lookup(item[0], 0)) #lookup(*,0)找key
            res["feats_val"].append(self.vocab.lookup(item[1], val_tpe))#lookup(*, 1)找value
        # label:即根据之前k-v对输出的语句，同样转换成词表中的语句, 此时的text lookup(*, 2)找词
        text = [self.vocab.lookup(word, 2) for word in line['desc'].split(" ")]
        # 句子长度
        slens = len(text)
        # 特征长度 即输入的长度
        res["feats_key_len"] = len(res["feats_key"])
        # 对应类型的字符串转换成id
        category = self.vocab.category2id[cate]
        # 前面加着sent 用于标识开始
        key_input = [self.vocab.lookup("<SENT>", 0)] + res['feats_key']         #前面加有sent的id的key
        val_input = [self.vocab.lookup("<ADJ>", val_tpe)] + res['feats_val']    #前面加有adj的id的value        这里adj必须属于词表中的value部分
        input_lens = len(key_input)                                             #输入的长度
        
        #一个样例有若干句子 所以这些都是list

        target_input = []   #target的输入:  开始符+seg转换成id         seg是对应的句子
        target_output = []  #输出：seg转换成id+结束符
        output_lens = []    #输出长度：输入输出都一样长，记录每个seg的长度

        '''
        关于group和target_type的区别，原始的数据集表示在描述一个物品时的一部分时，语句可能如group存的key-value对一样，
        描述裤型是阔腿裤，风格是性感之类的，但是语言的多样性导致实际情况风格可能描述为更个性化的语言，词表内没有对应词，
        如果组不成kv对，就单纯的不计入group，太过草率，所以用target_type记录描述的种类。
        即：target_type的值必存在词表中的种类key里
        但是group不一定有target_type值对应的key-value对
        构成数据集的情况十分复杂：
        针对target_type的值：
            1.句子中有词表中key部分的关键词，添加
            2.句子中有词表中value部分的关键词，添加对应的key
            3.两者均无，添加"GENERAL"
        group:句子中有value部分的关键词，添加对应key-value对
        '''
        group = []          #group: 记录每个seg中的key-value对的id形式
        group_lens = []     #group长度：记录group中每个seg有多少key-value

        target_type = []    #目标type：记录每个seg描述的种类，
        target_type_lens = []   #记录每个seg中有多少描述的种类

        key_val = list(zip(key_input, val_input))
        # line['segment']:字典  seg_0: {...} seg_1:{...}为k-v
        # seg_0:{...}其中{...}亦是一个字典 segID:int  key_type:key种类  order:key-value对  seg:生成的句子
        # _对seg_0  segment对{...}
        for _, segment in line['segment'].items():
            '''
            --parames:
                _:seg_0, seg_1。。。的字符串
                segment:
                    segID:int  表示第几个seg
                    key_type:string list 表示seg描述的种类，key-value中的key，但与当前的key-value对并非一一对应的
                    order:key-value对
                    seg:生成的句子
            '''
            sent = [self.vocab.lookup(w, 2) for w in segment['seg'].split(" ")] # sent 句子中的词的id列表
            target_output.append(sent + [self.vocab.end_token])                 # target_output 为sent + end符
            target_input.append([self.vocab.start_token] + sent)                # target_input 为start符+sent
            output_lens.append(len(target_output[-1]))                          # output长度targetoutput

            order = [item[:2] for item in segment['order']]                     # key-value对，必定只占前两位所以是item[:2]
            # order为0的时候即只存开始结束符
            if len(order) == 0:
                order = [['<SENT>', '<ADJ>']]
            # k,v string   
            gid = [key_val.index((self.vocab.lookup(k, 0), self.vocab.lookup(v, val_tpe))) for k, v in order]
            # group 是多个seg_i中 gid排序后的结果
            group.append(sorted(gid))
            # group_lens记录每个seg_i中gid的数量
            group_lens.append(len(group[-1]))
            # target_type记录gid中的key
            target_type.append([self.vocab.type2id[t] for t in segment['key_type']])
            # target_type_lens记录gid的长度
            target_type_lens.append(len(target_type[-1]))
        # group_cnt 记录多少个seg
        group_cnt = len(group)

        # 针对每个seg,判断哪个部分最长，然后补全长度。
        for item in [target_input, target_output, group, target_type]:
            max_len = -1
            for lst in item:
                max_len = max(max_len, len(lst))
            for idx, lst in enumerate(item):
                if len(lst) < max_len:
                    item[idx] = lst + [0] * (max_len - len(lst))

        '''
        --parames:
            key_input:整个句子中所有的key                                   sen_key_num*item    item:int                           
            val_input:整个句子中所有的value,与key_input同样长                sen_key_num*item    item:int      
            input_lens:是上述两个属性的长度                                 sen_key_num
            target_input:按seg分，将每个seg存下来，前面加上开始符               seg_num*item    item:list, (len(seg_i)+1)*int
            target_output:按seg分，将每个seg存下来，后面加上结束符              同上
            output_lens:按seg分，存每个seg的长度+1（因为加上了开始符和结束符）  seg_num*item    item:int    
            group:按seg分，将每个seg的key-value对存下，                         seg_num*item    item：int  为kv字典（该字典根据当前句子设置）中指定kv的序号
            group_lens:按seg分，记录每个seg有多少个key-value对                  seg_num*item    item:int
            group_cnt:记录多少个seg                                             seg_num
            target_type:按seg分，记录每个seg描述了什么类                         seg_num*item    item:list, seg_i_type_num*int
            target_type_lens:记录每个seg描述了多少类                            seg_numt*item   item:int
            text:整个句子的id形式                                               seq_len*item    item:int
            slens:整个句子的长度                                                seq_len
            category:属于什么大类                                               int
        '''
        return (
            np.array(key_input, dtype=np.int32), np.array(val_input, dtype=np.int32),
            np.array(input_lens, dtype=np.int32),
            np.array(target_input, dtype=np.int32), np.array(target_output, dtype=np.int32),
            np.array(output_lens, dtype=np.int32),
            np.array(group, dtype=np.int32), np.array(group_lens, dtype=np.int32),
            np.array(group_cnt, dtype=np.int32),
            np.array(target_type, dtype=np.int32), np.array(target_type_lens, dtype=np.int32),
            np.array(text, dtype=np.int32), np.array(slens, dtype=np.int32),
            np.array(category, dtype=np.int32),
        )

    def get_dataset(self, filename, train=True):
        def process(line):
            line = json.loads(line.decode())
            return self.process_inst(line)
        # 此时dataset是一行文本一行文本的格式
        dataset = tf.data.TextLineDataset(filename)
        # 首先map,即对dataset每个值进行map内的方法，对每个x执行后面 tf.py_func的操作
        # 用tf.py_func是为了让输出由np的数据格式直接转换成tensor
        # 序列化输入x，所以后面是一个列表, 这里据我理解是[x1] [x2]依次传入py_func进行运算的
        # Tout表示输出的14个值都转换成tf.int32的数据，
        # 这里我的理解是，转换tf.int32转换的是最根底的数据，这里很方便，因为原本就是int类型
        # 但是其实最后返回的是元组，所以，Tout=tf.int32亦可直接转换，只不过这时候dataset存的值由14...变为1*14...
        dataset = dataset.map(map_func=lambda x: tf.py_func(lambda y: process(y), [x], Tout=[tf.int32] * 14))
        #训练的时候打乱
        if train:
            dataset = dataset.shuffle(self.config.shuffle_buffer_size, reshuffle_each_iteration=True)
        
        def batching_func(x):
            #构建批次 
            #  batch_size*padded_shapes为数据集的格式  paddedshapes为上述一行的处理结果,里面每个item为int,每个None多一维
            return x.padded_batch(
                self.config.train_batch_size if (train) else self.config.test_batch_size,
                padded_shapes=(
                    tf.TensorShape([None]),
                    tf.TensorShape([None]),
                    tf.TensorShape([]),

                    tf.TensorShape([None, None]),
                    tf.TensorShape([None, None]),
                    tf.TensorShape([None]),

                    tf.TensorShape([None, None]),
                    tf.TensorShape([None]),
                    tf.TensorShape([]),

                    tf.TensorShape([None, None]),
                    tf.TensorShape([None]),

                    tf.TensorShape([None]),
                    tf.TensorShape([]),

                    tf.TensorShape([])
                )
            )
        # 选取最小的seg数量
        def key_func(p_1, p_2, input_len,
                     p_3, p_4, p_5,
                     p_6, p_7, gcnt,
                     p_8, p_9,
                     p_10, slen,
                     p_11
                     ):
            bucket_id = gcnt # slen // self.config.bucket_width
            return tf.to_int64(tf.minimum(self.config.num_buckets, bucket_id))

        def reduce_func(unused_key, windowed_data):
            return batching_func(windowed_data)

        if train:
            # 训练时对整个dataset执行tf.contrib.data.group_by_window操作
            # 作用是根据key_func，即group_cnt的大小，即句子分成几个seg进行处理，将数据分桶
            # 其实就是将group_cnt相同的数据存在一起，
            # reduce其实是因为有些数据无法构成一个batch，我们需要将其填充成一个batch
            dataset = dataset.apply(    
                tf.contrib.data.group_by_window(key_func=key_func, reduce_func=reduce_func,
                                                window_size=self.config.train_batch_size))
        else:
            dataset = batching_func(dataset)
        return dataset
