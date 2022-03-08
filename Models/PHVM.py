import tensorflow as tf
import numpy as np
import os
import sys
import collections
from Models import model_utils

class PHVMBatchInput(collections.namedtuple("PHVMBatchInput",
                                          ("key_input", "val_input", "input_lens",
                                           "target_input", "target_output", "output_lens",
                                           "group", "group_lens", "group_cnt",
                                           "target_type", "target_type_lens",
                                           "text", "slens",
                                           "category"))):
    pass

class PHVMConfig:
    def __init__(self):
        # rnn
        self.PHVM_rnn_direction = 'bi'
        self.PHVM_rnn_type = 'gru'

        # embedding
        self.share_vocab = False
        self.PHVM_word_dim = 300
        self.PHVM_key_dim = 30
        self.PHVM_val_dim = 100
        self.PHVM_cate_dim = 10

        # group
        self.PHVM_group_selection_threshold = 0.5
        self.PHVM_stop_threshold = 0.5
        self.PHVM_max_group_cnt = 30
        self.PHVM_max_sent_cnt = 10

        # type
        self.PHVM_use_type_info = False
        self.PHVM_type_dim = 30

        # encoder
        self.PHVM_encoder_dim = 100
        self.PHVM_encoder_num_layer = 1

        # group_decoder
        self.PHVM_group_decoder_dim = 100
        self.PHVM_group_decoder_num_layer = 1

        # group encoder
        self.PHVM_group_encoder_dim = 100
        self.PHVM_group_encoder_num_layer = 1

        # latent_decoder
        self.PHVM_latent_decoder_dim = 300
        self.PHVM_latent_decoder_num_layer = 1

        # sent_top_encoder
        self.PHVM_sent_top_encoder_dim = 300
        self.PHVM_sent_top_encoder_num_layer = 1

        # text post encoder
        self.PHVM_text_post_encoder_dim = 300
        self.PHVM_text_post_encoder_num_layer = 1

        # sent_post_encoder
        self.PHVM_sent_post_encoder_dim = 300
        self.PHVM_sent_post_encoder_num_layer = 1

        # bow
        self.PHVM_bow_hidden_dim = 200

        # decoder
        self.PHVM_decoder_dim = 300
        self.PHVM_decoder_num_layer = 2

        # latent
        self.PHVM_plan_latent_dim = 200
        self.PHVM_sent_latent_dim = 200

        # training
        self.PHVM_learning_rate = 0.001
        self.PHVM_num_training_step = 100000
        self.PHVM_sent_full_KL_step = 20000
        self.PHVM_plan_full_KL_step = 40000
        self.PHVM_dropout = 0

        # inference
        self.PHVM_beam_width = 10
        self.PHVM_maximum_iterations = 50

class PHVM:
    # 初始化各个的值
    
    def __init__(self, key_vocab_size, val_vocab_size, tgt_vocab_size, cate_vocab_size,
                 key_wordvec=None, val_wordvec=None, tgt_wordvec=None,
                 type_vocab_size=None, start_token=0, end_token=1, config=PHVMConfig()):
        '''
        key_vocab_size:输入的键值词表大小   len(dataset.vocab.id2featCate)
        val_vocab_size:输入的val值词表大小  len(dataset.vocab.id2featVal)
        tgt_vocab_size:输入的target词表的大小  即整个文本词表的大小  len(dataset.vocab.id2word)
        cate_vocab_size:大分类的词表的大小  len(dataset.vocab.id2category)
        type_vocab_size:输入的type类的大小  len(dataset.vocab.id2type)       type类与key近似，只是专属符不一样，详见Vocabulary.py20--35
        key_wordvec:暂为None main中亦未赋值
        val_wordvec:同上
        tgt_wordvec:dataset.vocab.id2vec
        start_token和end_token：均取默认值
        config:取上面的默认config
        '''
        self.config = config
        self.key_vocab_size = key_vocab_size    
        self.val_vocab_size = val_vocab_size    
        self.tgt_vocab_size = tgt_vocab_size    
        self.cate_vocab_size = cate_vocab_size  
        self.type_vocab_size = type_vocab_size  
        self.start_token = start_token
        self.end_token = end_token

        self.early_stopping = 15                #提前训练停止的次数

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input = self.get_input_tuple()                     #初始化图输入的占位
            self.build_graph(key_wordvec, val_wordvec, tgt_wordvec) #根据三个向量建图
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.init = tf.global_variables_initializer()
            self.sess = tf.Session(config=config)
            self.sess.run(self.init)
            self.best_saver = tf.train.Saver()
            self.tmp_saver = tf.train.Saver()
    # 获取对应占位 为输入获取对应占位
    def get_input_tuple(self):
        '''
        --parames:
            key_input:整个句子中所有的key                                   sen_key_num*item    item:int                           
            val_input:整个句子中所有的value,与key_input同样长                sen_key_num*item    item:int      
            input_lens:是上述两个属性的长度                                 sen_key_num
            target_input:按seg分，将每个seg存下来，前面加上开始符               seg_num*item    item:list, (len(seg_i)+1)*int
            target_output:按seg分，将每个seg存下来，后面加上结束符              同上
            output_lens:按seg分，存每个seg的长度+1（因为加上了开始符和结束符）  seg_num*item    item:int    
            group:按seg分，将每个seg的key-value对存下，                         seg_num*item    item:int  为指定kv对在字典表中的序号
            group_lens:按seg分，记录每个seg有多少个key-value对                  seg_num*item    item:int
            group_cnt:记录多少个seg                                             seg_num
            target_type:按seg分，记录每个seg描述了什么类                        seg_num*item    item:list, seg_i_type_num*int
            target_type_lens:记录每个seg描述了多少类                            seg_numt*item   item:int
            text:整个句子的id形式                                               seq_len*item    item:int
            slens:整个句子的长度                                                seq_len
            category:属于什么大类                                               int
        '''
        return PHVMBatchInput(

            key_input=tf.placeholder(shape=[None, None], dtype=tf.int32),
            val_input=tf.placeholder(shape=[None, None], dtype=tf.int32),
            input_lens=tf.placeholder(shape=[None], dtype=tf.int32),

            target_input=tf.placeholder(shape=[None, None, None], dtype=tf.int32),
            target_output=tf.placeholder(shape=[None, None, None], dtype=tf.int32),
            output_lens=tf.placeholder(shape=[None, None], dtype=tf.int32),

            group=tf.placeholder(shape=[None, None, None], dtype=tf.int32),
            group_lens=tf.placeholder(shape=[None, None], dtype=tf.int32),
            group_cnt=tf.placeholder(shape=[None], dtype=tf.int32),

            target_type=tf.placeholder(shape=[None, None, None], dtype=tf.int32),
            target_type_lens=tf.placeholder(shape=[None, None], dtype=tf.int32),

            text=tf.placeholder(shape=[None, None], dtype=tf.int32),
            slens=tf.placeholder(shape=[None], dtype=tf.int32),

            category=tf.placeholder(shape=[None], dtype=tf.int32)
        )
    # 获取学习率
    def get_learning_rate(self):
        self.learning_rate = tf.constant(self.config.PHVM_learning_rate, dtype=tf.float32)
        start_decay_step = self.config.PHVM_num_training_step // 2
        decay_times = 5
        decay_factor = 0.5
        decay_steps = (self.config.PHVM_num_training_step - start_decay_step) // decay_times
        return tf.cond(
            self.global_step < start_decay_step,        #全局步数<开始衰减的步数
            lambda: self.learning_rate,                 #学习率不变
            lambda: tf.train.exponential_decay(         #指数下降     
                self.learning_rate,                     #初始学习率
                tf.minimum(self.global_step - start_decay_step, 3 * decay_steps),   #用于计算的step值
                decay_steps, decay_factor, staircase=True),     #
            name="learning_rate_decay_cond")
    # 高斯采样
    def sample_gaussian(self, shape, mu, logvar):
        x = tf.random_normal(shape, dtype=tf.float32)   #根据shap随机生成高斯值
        return tf.cond(tf.equal(len(shape), 2),
                       lambda: mu + tf.exp(logvar / 2) * x,
                       lambda: tf.expand_dims(mu, 1) + tf.exp(tf.expand_dims(logvar / 2, 1)) * x)
    # embedding key value tgt embed的变量格式
    def make_embedding(self, key_wordvec, val_wordvec, tgt_wordvec):
        '''
        这里的操作是将已经训练好的词向量转换成常量tensor
        tgt_wordvec得到self.word_embedding: vocab_size*word_dim
        val_wordvec得到self.val_embedding
        key_wordvec得到self.key_embedding
        在无对应词表的情况下，随机初始化
        如果可共享此表，则val_embedding和word_embedding一致
        '''
        if tgt_wordvec is None:
            self.word_embedding = tf.get_variable("word_embedding",
                                                  shape=[self.tgt_vocab_size, self.config.PHVM_word_dim],
                                                  dtype=tf.float32)
        else:
            self.word_embedding = tf.get_variable("word_embedding", dtype=tf.float32,
                                                  initializer=tf.constant(tgt_wordvec, dtype=tf.float32))

        if self.config.share_vocab:
            self.val_embedding = self.word_embedding
        else:
            if val_wordvec is None:
                self.val_embedding = tf.get_variable("val_embedding",
                                                     shape=[self.val_vocab_size, self.config.PHVM_val_dim],
                                                     dtype=tf.float32)
            else:
                self.val_embedding = tf.get_variable("val_embedding", dtype=tf.float32,
                                                     initializer=tf.constant(val_wordvec, dtype=tf.float32))

        if key_wordvec is None:
            self.key_embedding = tf.get_variable("key_embedding",
                                                 shape=[self.key_vocab_size, self.config.PHVM_key_dim],
                                                 dtype=tf.float32)
        else:
            self.key_embedding = tf.get_variable("key_embedding", dtype=tf.float32,
                                                 initializer=tf.constant(key_wordvec, dtype=tf.float32))

        self.cate_embedding = tf.get_variable("cate_embedding",
                                              shape=[self.cate_vocab_size, self.config.PHVM_cate_dim],
                                              dtype=tf.float32)

        if self.config.PHVM_use_type_info:
            self.type_embedding = tf.get_variable("type_embedding",
                                                  shape=[self.type_vocab_size, self.config.PHVM_type_dim],
                                                  dtype=tf.float32)
    # 计算KL散度  需要的是隐层的均值和方差，实际的均值和方差 以及判断是否求均值
    def KL_divergence(self, prior_mu, prior_logvar, post_mu, post_logvar, reduce_mean=True):
        divergence = 0.5 * tf.reduce_sum(tf.exp(post_logvar - prior_logvar)
                                         + tf.pow(post_mu - prior_mu, 2) / tf.exp(prior_logvar)
                                         - 1 - (post_logvar - prior_logvar), axis=1)
        if reduce_mean:
            return tf.reduce_sum(divergence)
        else:
            return divergence
    # 针对某个句子的kv对的顺序，得到了最后句子句向量的表达，这里的信息只有kv对，seg划分的信息，无其他用于句子补全的信息
    def gather_group(self, src, group_idx, group_lens, group_cnt, group_encoder):
        '''
        --parames:
            src:        为下面src_encoder_output, [batch, n, encoder_dim*2]
            group_idx:  为输入的group [batch, seg_num, seg_kv_num]  item为int，当前句子的kv序号，是seg中的，但是序号是针对当前句子设置的
                            seg_num为某句话的seg数，seg_kv_num为seg的kv数
            group_lens：[batch, seg_num] item为int，为seg_kv_num的值
            group_cnt： [batch,]         item为int，为seg_num的值
            group_encoder:  为一循环层， group_encoder_num_layer*group_encoder_dim大小，输出单元为group_encoder_dim
        值得注意的是，当seg_num等为矩阵中的某一维时，大小是seg_num中最大的那个，从而对齐数据，为了使数据更精确，才需记录实际的seg_num等。
        即当为矩阵的一维时，值为max，只有当其为item时，才是每个句子实际的值
        --returns:
            gidx:   [batch, seg_num, seg_kv_num, 2]   指定句子的seg中的某个kv的编号 
            group_bow:  [batch, seg_num. seg_kvnum, encoder_dim*2]  指定句子的seg中的某个kv的encoder形式
            group_mean_bow: [batch, seg_num, encoder_dim*2]         指定句子中的seg的编码形式
            group_embed:[batch, group_encoder_dim]                  指定句子中的句向量表达
        '''
        
        shape = tf.shape(group_idx) # 值为[batch, seg_num, seg_kv_num,2 ] 
        batch_size = shape[0]
        fidx = tf.expand_dims(tf.expand_dims(tf.range(batch_size), 1), 2) # [batch,1,1]
        fidx = tf.tile(fidx, [1, shape[1], shape[2]])   # [batch, seg_num, seg_kv_num] 复制扩展
        fidx = tf.expand_dims(fidx, 3)                  # [batch, seg_num,seg_kv_num, 1] 值是0--batch-1的int
        sidx = tf.expand_dims(group_idx, 3)             # [batch, seg_num,seg_kv_num, 1] item是int 指定kv序号
        gidx = tf.concat((fidx, sidx), 3)               # [batch, seg_num, seg_kv_num, 2]   有batch的编号和kv的序号 当前句子
        # gidx指定了当前句子的序号，以及对应kv对的序号
        # src指定了当前句子的kv对的编码形式
        group_bow = tf.gather_nd(src, gidx)             # [batch, seg_num. seg_kvnum, encoder_dim*2]    把对应kv的编码形式得到    
        # 根据group_lens添加每个句子的每个seg_kv_num值，得到mask，用于防止后面的空数据影响  [batch, seg_num, seg_kv_num] 值为1.0/0.0        
        group_mask = tf.sequence_mask(group_lens, shape[2], dtype=tf.float32)  
        expanded_group_mask = tf.expand_dims(group_mask, 3) # [batch, seg_num, seg_kv_num, 1]
        group_sum_bow = tf.reduce_sum(group_bow * expanded_group_mask, 2)   # [batch, seg_num, encoder_dim*2] 将kv编码的
        safe_group_lens = group_lens + tf.cast(tf.equal(group_lens, 0), dtype=tf.int32) 
        # 同之前操作一直，[batch, seg_num, encoder_dim*2]  将kv对的编码的均值作为句子中seg的kv编码结果
        group_mean_bow = group_sum_bow / tf.to_float(tf.expand_dims(safe_group_lens, 2))    
        # 用group_encoder作为循环层
        # group_mean_bow作为输入  其中seg_num表示time_step
        # group_encoder_output [batch, seg_num, group_encoder_dim]
        # group_encoder_state   [batch, group_encoder_dim]
        group_encoder_output, group_encoder_state = tf.nn.dynamic_rnn(group_encoder,
                                                                      group_mean_bow,
                                                                      # 输入的最大长度 即time_step的最大长度，这里是seg的最大数量 [batch,]
                                                                      group_cnt,        
                                                                      dtype=tf.float32)
        # group_embed [batch, group_encoder_dim]
        if self.config.PHVM_rnn_type == 'lstm':
            group_embed = group_encoder_state.h
        else:
            group_embed = group_encoder_state
        return gidx, group_bow, group_mean_bow, group_embed
    # 根据输入的向量建图，这里的向量并非是输入的句子向量，而是字典中训练好的词向量
    def build_graph(self, key_wordvec, val_wordvec, tgt_wordvec):
        '''
        --parames:
            key_wordvec:类的词向量字典
            val_wordvec:值的词向量字典
            tgt_wordvec:所有词的字典
        这里main只输入了tgt_wordvec
        '''
        self.global_step = tf.get_variable("global_step", dtype=tf.int32, initializer=tf.constant(0), trainable=False)
        self.keep_prob = tf.placeholder(shape=[], dtype=tf.float32)
        self.train_flag = tf.placeholder(shape=[], dtype=tf.bool)
        self.batch_size = tf.size(self.input.input_lens)
        
        # embedding 的图 得到k-v的embed(src) 构建了cate_embed 根据是否使用例参数
        # 此处embed不止 k-v 还有target的text的embed
        '''
            key_embed:[batch,kv_num,key_dim] kv_num是总计词表有多少个kv
            val_embed:[batch,kv_num,val_dim] 
            src:[batch, kv_num, key_dim+val_dim] 
            cate_embed:[batch, cate_dim]
            type_embed:[batch, seg_num, type_dim] 每个seg表示的向量为seg内每个类的向量的均值
            text:[batch, text_len, word_dim]
        '''
        with tf.variable_scope("embedding"):
            self.make_embedding(key_wordvec, val_wordvec, tgt_wordvec)
            # key_embed:通过key_embedding,key_input是序列化数据，input中的一条:[batch,n,]->[batch,n,key_dim]
            key_embed = tf.nn.embedding_lookup(self.key_embedding, self.input.key_input)  
            # 同理[batch,n,]->[batch,n,val_dim]
            val_embed = tf.nn.embedding_lookup(self.val_embedding, self.input.val_input)
            # src[batch, n, key_dim+val_dim]
            src = tf.concat((key_embed, val_embed), 2)  
            # cate_embed[batch, cate_dim]
            cate_embed = tf.nn.embedding_lookup(self.cate_embedding, self.input.category)
            # 使用type这个属性的时候
            if self.config.PHVM_use_type_info:
                #type_embed  [batch, seg_num, seg_type_num, type_dim]  批次  一句话几个seg  一个seg描述了几个属性  属性的向量纬度
                type_embed = tf.nn.embedding_lookup(self.type_embedding, self.input.target_type)
                # type_mask:[batch, seg_num, seg_type_num]的float型 
                # 将前面的为1，后面的补0，这样的mask得到有效部分为1，无效部分为0
                type_mask = tf.sequence_mask(self.input.target_type_lens,               # [batch, seg_num]
                                             tf.shape(self.input.target_type)[2],       # [batch, seg_num, seg_type_num] 即seg_type_num
                                             dtype=tf.float32)
                # [batch, seg_num, seg_type_num, 1]
                expanded_type_mask = tf.expand_dims(type_mask, 3)
                # type_embed * expanded_type_mask得到 mask后的 type_embed [batch, seg_num, seg_type_num, type_dim]
                # type_embed [batch, seg_num, type_dim]     这里将每个seg的type相加作为seg的type vec
                type_embed = tf.reduce_sum(type_embed * expanded_type_mask, 2)
                # 将target_type_lens: [batch, seg_num]
                # 将其中描述的类数为0的加1，使得下面做除法正常
                safe_target_type_lens = self.input.target_type_lens + \
                                        tf.cast(tf.equal(self.input.target_type_lens, 0), dtype=tf.int32)
                # 刚刚是和作为type vec，现在转换成均值作为type vec
                type_embed = type_embed / tf.cast(tf.expand_dims(safe_target_type_lens, 2), dtype=tf.float32)
            # 把文本转换成对应的embed 形式
            # text [batch, text_len, word_dim]
            text = tf.nn.embedding_lookup(self.word_embedding, self.input.text)
        
        # input_encode 的图  输入是k-v对
        '''
            输入为src:[batch, kv_num, key_dim+val_dim]
            输出为src_embed: [batch, encoder_dim(*2)] 
        '''
        with tf.variable_scope("input_encode"):
            # 是否双向
            if self.config.PHVM_rnn_direction == 'uni':
                # 单向
                src_cell = model_utils.get_rnn_cell(self.config.PHVM_rnn_type,
                                                    self.config.PHVM_encoder_num_layer,
                                                    self.config.PHVM_encoder_dim,
                                                    self.keep_prob,
                                                    "src_encoder")
                # src_encoder_output:[batch, time,encoder_dim]
                # src_encoder_state:[batch, encoder_dim]
                src_encoder_output, src_encoder_state = tf.nn.dynamic_rnn(src_cell,
                                                                          src,  #为上述k-v连接成的结果 [batch, n, key_dim+val_dim]
                                                                          self.input.input_lens,
                                                                          dtype=tf.float32)
                # src_embed: [batch, encoder_dim] 
                if self.config.PHVM_rnn_type == 'lstm':
                    src_embed = src_encoder_state.h                          
                else:
                    src_embed = src_encoder_state
            else:
                # 根据参数，构建gru层 得出编码层的输出和state 输出为[h_n-> , <-h_1] state为i个神经元的[h_i->,<-h_i]
                src_fw_cell = model_utils.get_rnn_cell(self.config.PHVM_rnn_type,
                                                       self.config.PHVM_encoder_num_layer,
                                                       self.config.PHVM_encoder_dim,
                                                       self.keep_prob,
                                                       "src_fw_encoder")
                src_bw_cell = model_utils.get_rnn_cell(self.config.PHVM_rnn_type,
                                                       self.config.PHVM_encoder_num_layer,
                                                       self.config.PHVM_encoder_dim,
                                                       self.keep_prob,
                                                       "src_bw_encoder")
                # src_encoder_output:[2,batch, kv_num, encoder_dim]   n为句子的长度 这里是kv对的最大数目
                src_encoder_output, src_encoder_state = tf.nn.bidirectional_dynamic_rnn(src_fw_cell,
                                                                                        src_bw_cell,
                                                                                        src,
                                                                                        self.input.input_lens,
                                                                                        dtype=tf.float32)
                src_encoder_output = tf.concat(src_encoder_output, 2)
                # src_embed: [batch, encoder_dim*2] 
                if self.config.PHVM_rnn_type == 'lstm':
                    # 01分别代表正向逆向输出，[[h1z,h2z,...],[h1n,h2n,...]]->[(h1z,h1n), (h2z,h2n)...]   
                    # 2*n*h -> n*2h   n为输入的向量长
                    src_embed = tf.concat((src_encoder_state[0].h, src_encoder_state[1].h), 1)
                else:
                    src_embed = tf.concat((src_encoder_state[0], src_encoder_state[1]), 1)
        
        # 文本的编码 
        '''
            输入为text: [batch, text_len, word_dim]
            输出为tgt_embed:[batch, text_post_encoder_dim*2]
        '''
        with tf.variable_scope("text_encode"):
            tgt_fw_cell = model_utils.get_rnn_cell(self.config.PHVM_rnn_type,
                                                   self.config.PHVM_text_post_encoder_num_layer,
                                                   self.config.PHVM_text_post_encoder_dim,
                                                   self.keep_prob,
                                                   "text_fw_encoder")
            tgt_bw_cell = model_utils.get_rnn_cell(self.config.PHVM_rnn_type,
                                                   self.config.PHVM_text_post_encoder_num_layer,
                                                   self.config.PHVM_text_post_encoder_dim,
                                                   self.keep_prob,
                                                   "text_bw_encoder")
            tgt_encoder_output, tgt_encoder_state = tf.nn.bidirectional_dynamic_rnn(tgt_fw_cell,
                                                                                    tgt_bw_cell,
                                                                                    text,
                                                                                    self.input.slens,
                                                                                    dtype=tf.float32)
            if self.config.PHVM_rnn_type == 'lstm':
                tgt_embed = tf.concat((tgt_encoder_state[0].h, tgt_encoder_state[1].h), 1)
            else:
                tgt_embed = tf.concat((tgt_encoder_state[0], tgt_encoder_state[1]), 1)
        
        # 利用cate, src_embed（key_embed, val_embed）, tgt_embed（text_post_encoder_dim）得到dec_input
        '''
            输入为：cate_embed:[batch, cate_dim]                    大类
                src_embed: [batch, encoder_dim(*2)]                 kv编码
                tgt_embed:[batch, text_post_encoder_dim*2]          文本句子编码
            输出为：dec_input:[batch, cate_dim + encoder_dim(*2) + plan_latent_dim]       
                根据是否训练，重采样时用的隐含变量不一样 
        '''
        with tf.variable_scope("top_level"):
            # 重采样
            '''
            输入为prior_input：[batch, cate_dim+encoder_dim(*2)]   tf.concat((cate_embed, src_embed), 1)
            输出为prior_z_plan：[batch, plan_latent_dim]        prior_mu和prior_logvar构成
            '''
            with tf.variable_scope("prior_network"):
                # prior_input [batch, cate_dim+encoder_dim(*2)]
                prior_input = tf.concat((cate_embed, src_embed), 1)
                prior_fc = tf.layers.dense(prior_input, self.config.PHVM_plan_latent_dim * 2, activation=tf.tanh)
                #prior_fc_nd:[batch, plan_latent_dim*2]
                prior_fc_nd = tf.layers.dense(prior_fc, self.config.PHVM_plan_latent_dim * 2)
                # prior_mu/prior_logvar:[batch, plan_latent_dim]
                prior_mu, prior_logvar = tf.split(prior_fc_nd, 2, 1)    #在1维切成两份
                # prior_z_plan:[batch, plan_latent_dim]
                prior_z_plan = self.sample_gaussian((self.batch_size, self.config.PHVM_plan_latent_dim),
                                                    prior_mu,
                                                    prior_logvar)
            # 加上目标重采样
            '''
            输入为post_input：[batch, cate_dim+encoder_dim(*2)+text_post_encoder_dim*2]   tf.concat((cate_embed, src_embed, tgt_embed), 1)
            输出为post_z_plan：[batch,plan_latent_dim] post_mu和post_logvar构成
            '''
            with tf.variable_scope("posterior_network"):
                #post_input:[batch, cate_dim+encoder_dim(*2)+text_post_encoder_dim*2]
                post_input = tf.concat((cate_embed, src_embed, tgt_embed), 1)
                post_fc = tf.layers.dense(post_input, self.config.PHVM_plan_latent_dim * 2)
                post_mu, post_logvar = tf.split(post_fc, 2, 1)
                #post_z_plan:[batch,plan_latent_dim]
                post_z_plan = self.sample_gaussian((self.batch_size, self.config.PHVM_plan_latent_dim),
                                                   post_mu,
                                                   post_logvar)
            # 判断两个采样得到的KL散度
            # plan_KL_divergence:[batch,]/int       取决于reduce_mean的属性
            self.plan_KL_divergence = self.KL_divergence(prior_mu, prior_logvar, post_mu, post_logvar)
            plan_KL_weight = tf.minimum(1.0, tf.to_float(self.global_step) / self.config.PHVM_plan_full_KL_step)
            # 是否处于训练，训练时下一步输入为加上目标的重采样，不训练时即无目标的重采样
            # 类embed,encode_embed, post_z/prior_z（上面两种不同输入的重采样）
            # dec_input:[batch, cate_dim + encoder_dim(*2) + plan_latent_dim]       根据是否训练，重采样时用的隐含变量不一样
            dec_input = tf.cond(self.train_flag,
                                lambda: tf.concat((cate_embed, src_embed, post_z_plan), 1),
                                lambda: tf.concat((cate_embed, src_embed, prior_z_plan), 1))
            # 记录dec_input的纬度，如上述缩写
            if self.config.PHVM_rnn_direction == 'uni':
                dec_input_dim = self.config.PHVM_cate_dim + self.config.PHVM_encoder_dim + \
                                self.config.PHVM_plan_latent_dim
            else:
                dec_input_dim = self.config.PHVM_cate_dim + 2 * self.config.PHVM_encoder_dim + \
                                self.config.PHVM_plan_latent_dim
            dec_input = tf.reshape(dec_input, [self.batch_size, dec_input_dim])
        
        # 句子级
        with tf.variable_scope("sentence_level"):
            # 构建了4个循环层  无执行输入
            with tf.variable_scope("group_sent"):
                group_decoder = model_utils.get_rnn_cell(self.config.PHVM_rnn_type,
                                                         self.config.PHVM_group_decoder_num_layer,
                                                         self.config.PHVM_group_decoder_dim,
                                                         self.keep_prob,
                                                         'group_decoder')

                group_encoder = model_utils.get_rnn_cell(self.config.PHVM_rnn_type,
                                                         self.config.PHVM_group_encoder_num_layer,
                                                         self.config.PHVM_group_encoder_dim,
                                                         self.keep_prob,
                                                         "group_encoder")
                # 隐变量输入
                latent_decoder = model_utils.get_rnn_cell(self.config.PHVM_rnn_type,
                                                          self.config.PHVM_latent_decoder_num_layer,
                                                          self.config.PHVM_latent_decoder_dim,
                                                          self.keep_prob,
                                                          "latent_decoder")

                decoder = model_utils.get_rnn_cell(self.config.PHVM_rnn_type,
                                                   self.config.PHVM_decoder_num_layer,
                                                   self.config.PHVM_decoder_dim,
                                                   self.keep_prob,
                                                   "decoder")
            # 构建了一个双向循环层 无执行输入
            with tf.variable_scope("sent_post_encoder"):
                sent_fw_cell = model_utils.get_rnn_cell(self.config.PHVM_rnn_type,
                                                        self.config.PHVM_sent_post_encoder_num_layer,
                                                        self.config.PHVM_sent_post_encoder_dim,
                                                        self.keep_prob,
                                                        "sent_fw_cell")
                sent_bw_cell = model_utils.get_rnn_cell(self.config.PHVM_rnn_type,
                                                        self.config.PHVM_sent_post_encoder_num_layer,
                                                        self.config.PHVM_sent_post_encoder_dim,
                                                        self.keep_prob,
                                                        "sent_bw_cell")
            # 构建了若干全连接层
            with tf.variable_scope("parameters"):
                # group_init_state_fc: group_decoder_dim
                # init_gbow:encoder_dim(*2)
                # plan_init_state_fc:latent_decoder_dim
                with tf.variable_scope("init_state"):
                    group_init_state_fc = tf.layers.Dense(self.config.PHVM_group_decoder_dim)
                    if self.config.PHVM_rnn_direction == 'uni':
                        init_gbow = tf.get_variable("start_of_group", dtype=tf.float32,
                                                    shape=(1, self.config.PHVM_encoder_dim))
                    else:
                        init_gbow = tf.get_variable("start_of_group", dtype=tf.float32,
                                                   shape=(1, 2 * self.config.PHVM_encoder_dim))
                    plan_init_state_fc = tf.layers.Dense(self.config.PHVM_latent_decoder_dim)

                # prior_fc_layer/post_fc_layer:sent_latent_dim * 2
                prior_fc_layer = tf.layers.Dense(self.config.PHVM_sent_latent_dim * 2)
                post_fc_layer = tf.layers.Dense(self.config.PHVM_sent_latent_dim * 2)
                # group_fc_1:encoder_dim    group_fc_2:1
                group_fc_1 = tf.layers.Dense(self.config.PHVM_encoder_dim)
                group_fc_2 = tf.layers.Dense(1)
                # type_fc_1:type_dim        type_fc_2:type_vocab_size
                type_fc_1 = tf.layers.Dense(self.config.PHVM_type_dim)
                type_fc_2 = tf.layers.Dense(self.type_vocab_size)
                # bow_fc_1:bow_hidden_dim   bow_fc_2:tgt_vocab_size
                bow_fc_1 = tf.layers.Dense(self.config.PHVM_bow_hidden_dim)
                bow_fc_2 = tf.layers.Dense(self.tgt_vocab_size)
                # projection:tgt_vocab_size
                projection = tf.layers.Dense(self.tgt_vocab_size)
                # stop_clf:1
                stop_clf = tf.layers.Dense(1)
            # 训练 包含梯度更新
            with tf.name_scope("train"):
                '''
                    gidx:   [batch, seg_num, seg_kv_num, 2]                 指定句子的seg中的某个kv的编号 
                    group_bow:  [batch, seg_num. seg_kvnum, encoder_dim*2]  指定句子的seg中的某个kv的encoder形式
                    group_mean_bow: [batch, seg_num, encoder_dim*2]         指定句子中的seg的编码形式
                    group_embed:[batch, group_encoder_dim]                  指定句子中的句向量表达 kv所属哪个seg的信息
                '''
                with tf.name_scope("group_encode"):
                    gidx, group_bow, group_mean_bow, group_embed = self.gather_group(src_encoder_output,    # 输入的encoder输出
                                                                                     self.input.group,      # group属性 
                                                                                     self.input.group_lens,
                                                                                     self.input.group_cnt,
                                                                                     group_encoder)         # group编码
                # 判断i是否比句子的长度大  i<seq_len:true 
                def train_cond(i, group_state, gbow, plan_state, sent_state, sent_z,
                               stop_sign, sent_rec_loss, group_rec_loss, KL_loss, type_loss, bow_loss):
                    return i < tf.shape(self.input.output_lens)[1] # 句子seg数比较
                
                # 依次对指定句子的seg进行处理，得到指定句子的损失 重点返回有：.
                # group_state:      主要包含句子类，kv对的信息，训练时还有句子本身的信息, 不断根据seg信息的变化而更新
                # plan_state:       包含有句子大类的信息，kv对的信息，以及句子本身的信息该句子的句向量表达  kv所属哪个seg的信息
                # sent_state:       传向下一步的状态,值包含attention的s2s decoder的最后一层状态
                # sent_rec_loss:    float 计算生成句子的loss， 为当前句子第i个seg的句子,经历decoder得到的
                # group_rec_loss:   判断该seg有哪些kv对的损失
                # KL_loss:          每个seg KL损失的累加 KL的两个分布均有当前得到的plan信息，当前句子当前seg的kv对信息，
                #                   post还有当前句子当前seg的编码信息和当前句子的大类信息
                # type_loss:        计算类型的loss， 类型是当前句子第i个seg描述了什么类
                # bow_loss:         计算当前生成句子第i个seg的句子对应的词的损失，由dense得到，未经历decoder
                def train_body(i, group_state, gbow, plan_state, sent_state, sent_z,
                               stop_sign, sent_rec_loss, group_rec_loss, KL_loss, type_loss, bow_loss):
                    '''
                    本函数的训练在于对每个句子的第i个seg进行处理，得到指定的参数
                    --parames:初始化时   
                        i:              循环次数
                        group_state:    [batch, group_decoder_dim]    
                                        主要包含句子类，kv对的信息，训练时还有句子本身的信息
                        gbow:           [batch, encoder_dim(*2)]     init_gbow是随机生成的长encoder_dim(*2)的变量
                        plan_state:     [batch, latent_decoder_dim]  包含有句子大类的信息，kv对的信息，以及句子本身的信息
                                        该句子的句向量表达  kv所属哪个seg的信息
                        sent_state:     取decoder的最后一层为sent_state   大小为[decoder_dim,]
                        sent_z:         [batch_size, sent_latent_dim] 的0值tensor
                        stop_sign:      [seg_num, batch]
                        sent_rec_loss:  float 计算生成句子的loss， 为当前句子第i个seg的句子,经历decoder得到的
                        group_rec_loss: 判断该seg有哪些kv对的损失
                        KL_loss:        每个seg KL损失的累加 KL的两个分布均有当前得到的plan信息，当前句子当前seg的kv对信息，
                                        post还有当前句子当前seg的编码信息和当前句子的大类信息
                        type_loss:      计算类型的loss， 类型是当前句子第i个seg描述了什么类
                        bow_loss:       计算当前生成句子第i个seg的句子对应的词的损失，由dense得到，未经历decoder
                    上述参数均为循环体的参数, 
                    '''
                    # 第i个seg 
                    # sent_input:   [batch, seg_word_num, word_dim]
                    sent_input = tf.nn.embedding_lookup(self.word_embedding, self.input.target_input[:, i, :])
                    # sent_output:  [batch, seg_word_num]
                    sent_output = self.input.target_output[:, i, :]
                    # sent_lens:    [batch,]    值为第i个seg的词数
                    sent_lens = self.input.output_lens[:, i]
                    # sent_mask:    [batch, seg_word_num]   值为sent_lens构成的mask 为对应每个句子第i个seg的实际seg_word_num
                    sent_mask = tf.sequence_mask(sent_lens, tf.shape(sent_output)[1], dtype=tf.float32)
                    # loss_mask:    [batch,]   
                    loss_mask = 1 - tf.to_float(tf.equal(sent_lens, 0))
                    # 计算batch中每个句子对应的seg共有多少个  求对应的空值
                    effective_cnt = tf.to_float(tf.reduce_sum(1 - tf.to_int32(tf.equal(sent_lens, 0))))
                    '''
                        输入sent_input:[batch, seg_word_num, word_dim]
                        输出sent_embed:[batch, sent_post_encoder_dim*2] 当前seg的句子的编码
                    '''
                    with tf.variable_scope("sent_encoder"):
                        
                        # cali_sent_lens: 所有sent_lens的值-1，0不变
                        cali_sent_lens = sent_lens - 1 + tf.cast(tf.equal(sent_lens, 0), dtype=tf.int32)
                        # sent_encoder_output:[2, batch, seg_word_num, sent_post_encoder_dim]  seg_word_num 为time_steps
                        # sent_encoder_state: [2, batch, sent_post_encoder_dim] 
                        sent_encoder_output, sent_encoder_state = tf.nn.bidirectional_dynamic_rnn(sent_fw_cell,
                                                                                                  sent_bw_cell,
                                                                                                  sent_input[:, 1:, :], #input有开始符，所以从第二个开始
                                                                                                  cali_sent_lens,
                                                                                                  dtype=tf.float32)
                        # sent_embed:[batch, sent_post_encoder_dim*2]
                        if self.config.PHVM_rnn_type == 'lstm':
                            sent_embed = tf.concat((sent_encoder_state[0].h, sent_encoder_state[1].h), 1)
                        else:
                            sent_embed = tf.concat((sent_encoder_state[0], sent_encoder_state[1]), 1)
                    '''
                    输入:
                        gidx:   [batch, seg_num, seg_kv_num, 2]                 指定句子的seg中的某个kv的编号 
                        group_bow:  [batch, seg_num. seg_kvnum, encoder_dim*2]  指定句子的seg中的某个kv的encoder形式
                        src_encoder_output:[batch, kv_num, encoder_dim*2]       指定句子的所有kv对的encoder形式
                    输出:
                        group_rec_loss:[batch,]     指定句子的第i个seg中，kv对分布的损失，decoder得到的是kv在该seg的概率，然后与实际值做交叉熵
                    更新:
                        gbow:[batch, encoder_dim]   指定句子中第i个seg的所有kv对综合在一起的编码形式
                        stop_sign:[seg_num, batch]           根据当前seg的输入和decoder的上一状态解码得到的输出gout，
                    '''
                    with tf.name_scope("group"):
                        # sent_gid:[batch, seg_kv_num, 2]    指定句子的第i个seg中的kv在当前句子的编号
                        sent_gid = gidx[:, i, :, :]
                        # sent_group:[batch, seg_kv_num, encoder_dim*2] 指定句子的第i个seg中的kv的encoder形式
                        sent_group = group_bow[:, i, :, :]
                        # sent_group_len:[batch,]   指定句子第i个seg中有几个kv对
                        sent_group_len = self.input.group_lens[:, i]
                        # 防止÷0报错 [batch,]
                        safe_sent_group_len = sent_group_len + tf.cast(tf.equal(sent_group_len, 0), dtype=tf.int32)
                        # group_mask:[batch, seg_kv_num] 把实际的指定句子第i个kv的数量使用mask，得到更精确信息 
                        group_mask = tf.sequence_mask(sent_group_len, tf.shape(sent_group)[1], dtype=tf.float32)
                        # expanded_group_mask:[batch, seg_kv_num, 1]
                        expanded_group_mask = tf.expand_dims(group_mask, 2)
                        '''
                            根据随机生成的数作为初始化的输入，在输入指定句子第i个seg的编码后利用利用dense得到值作为初始化decoder的上一状态,
                            然后利用group_decoder得到指定解码，将得到的解码构成针对指定句子第i个seg的kv分布
                            即每个句子构成了kv的list, 针对该seg判断有哪些kv入内，
                            得到group_rec_loss:[batch,]   
                        '''
                        with tf.variable_scope("decode_group"):
                            # gout:         [batch, group_decoder_dim]
                            # group_state:  [batch, group_decoder_dim]
                            gout, group_state = group_decoder(gbow, group_state)    # 某一刻的输入，前一刻的输出，得到当前时刻的输出
                            # tile_gout:   [batch, kv_num, group_decoder_dim]   扩充时间维度
                            tile_gout = tf.tile(tf.expand_dims(gout, 1), [1, tf.shape(src_encoder_output)[1], 1])
                            # group_fc_input:   [batch, kv_num, group_decoder_dim + encoder_dim*2]
                            group_fc_input = tf.concat((src_encoder_output, tile_gout), 2)
                            #               [batch, kv_num, group_decoder_dim + encoder_dim*2]->     group_fc_1
                            #               [batch, kv_num, encoder_dim]->                           group_fc_2   tf.squeeze()
                            # group_logit:  [batch, kv_num]                                          神经网络计算得到
                            group_logit = tf.squeeze(group_fc_2(tf.tanh(group_fc_1(group_fc_input))), 2)
                            # group_label:[batch, seg_kv_num, kv_num]   把指定句子第i个seg的每个kv对序号转成one-hot格式  由输入决定
                            group_label = tf.one_hot(sent_gid[:, :, 1], tf.shape(group_logit)[1], dtype=tf.float32)
                            # group_label:[batch, kv_num]   mask指定句子，第i个seg多出来的Kv为0. 针对第i个seg,得到的值
                            group_label = tf.reduce_sum(group_label * expanded_group_mask, 1)
                            # group_crossent:[batch, kv_num]求交叉熵 是逻辑损失。理论上，每个句子的kv 
                            group_crossent = tf.nn.sigmoid_cross_entropy_with_logits(labels=group_label,
                                                                                     logits=group_logit)
                            # src_mask:[batch, kv_num] mask输入为空的kv对部分
                            src_mask = tf.sequence_mask(self.input.input_lens, tf.shape(group_logit)[1],
                                                        dtype=tf.float32)
                            group_crossent = loss_mask * tf.reduce_sum(group_crossent * src_mask, 1)
                            # group_rec_loss:[batch,]   
                            group_rec_loss += tf.reduce_sum(group_crossent) # / effective_cnt

                        # gbow:[batch, encoder_dim] 指定句子中第i个seg的所有kv对结合的编码形式
                        gbow = group_mean_bow[:, i, :]
                        # 记录stop_sign:[seg_num, batch] 写到第i个seg里
                        with tf.name_scope("stop_loss"):
                            stop_sign = stop_sign.write(i, tf.squeeze(stop_clf(gout), axis=1))

                    if self.config.PHVM_use_type_info:
                        # type 相当于seg中kv对的k的含义，但与句子的kv对中的k并不一定一致
                        sent_type_embed = type_embed[:, i, :]               #[batch, type_dim]          指定句子第i个seg的type编码
                        sent_type = self.input.target_type[:, i, :]         #[batch, seg_i_type_num]    指定句子中第i个seg都有哪些type
                        sent_type_len = self.input.target_type_lens[:, i]   #[batch,]
                        sent_type_mask = tf.sequence_mask(sent_type_len, tf.shape(sent_type)[1], dtype=tf.float32)  #[batch,seg_i_type_num]

                    # latent_decoder_input 
                    # plan_input:[batch, decoder_dim+sent_latent_dim]   初始均无信息
                    if self.config.PHVM_rnn_type == 'lstm':
                        plan_input = tf.concat((sent_state.h, sent_z), 1)
                    else:
                        plan_input = tf.concat((sent_state, sent_z), 1)
                    # plan_state初始化含句子大类和kv对的信息
                    # plan_input取决于上一步的状态，初始化为0
                    # sent_cond_embed/plan_state:[batch, latent_decoder_dim]    
                    sent_cond_embed, plan_state = latent_decoder(plan_input, plan_state)
                    
                    '''
                    输入:
                        sent_cond_embed:[batch, latent_decoder_dim] 上一步的plan信息，通过上一步的sent的相关状态和当前句子kv对所构建的信息
                        gbow:           [batch, encoder_dim]    指定句子指定seg的kv信息
                    输出:
                        sent_prior_mu/sent_prior_logvar:[batch, sent_latent_dim]   得到sent的隐含变量
                    '''
                    with tf.name_scope("sent_prior_network"):
                        # sent_prior_input:[batch, latent_decoder_dim + encoder_dim]
                        sent_prior_input = tf.concat((sent_cond_embed, gbow), 1) #指定句子指定seg的所有kv对的编码信息，上一步的plan信息
                        sent_prior_fc = prior_fc_layer(sent_prior_input)
                        # sent_prior_mu/sent_prior_logvar:[batch, sent_latent_dim]   得到隐含变量
                        sent_prior_mu, sent_prior_logvar = tf.split(sent_prior_fc, 2, axis=1)
                    
                    '''
                    输入:
                        sent_cond_embed:[batch, latent_decoder_dim] 上一步的plan信息，通过上一步的sent的相关状态和当前句子kv对所构建的信息
                        gbow:           [batch, encoder_dim]    指定句子指定seg的kv信息
                        sent_embed:     [batch, sent_post_encoder_dim*2] 指定句子指定seg的文本的编码
                        sent_type_embed:[batch, type_dim]       指定句子指定seg的type信息
                    输出:
                        sent_z:         [batch, sent_latent_dim] 通过上述信息输出的隐变量
                    '''
                    with tf.name_scope("sent_posterior_network"):
                        # sent_post_input:[batch, latent_decoder_dim+encoder_dim+sent_post_encoder_dim*2(+type_dim)]
                        if self.config.PHVM_use_type_info:
                            sent_post_input = tf.concat((sent_cond_embed, gbow, sent_embed, sent_type_embed), 1)
                        else:
                            sent_post_input = tf.concat((sent_cond_embed, gbow, sent_embed), 1)
                        sent_post_fc = post_fc_layer(sent_post_input)
                        # sent_post_mu/sent_post_logvar:[batch, sent_latent_dim]
                        sent_post_mu, sent_post_logvar = tf.split(sent_post_fc, 2, axis=1)
                        sent_z = self.sample_gaussian((self.batch_size, self.config.PHVM_sent_latent_dim),
                                                      sent_post_mu,
                                                      sent_post_logvar)
                        # sent_z:[batch, sent_latent_dim]
                        sent_z = tf.reshape(sent_z, (self.batch_size, self.config.PHVM_sent_latent_dim))

                    # sent_cond_z_embed:[batch, latent_decoder_dim+sent_latent_dim] 上一步的plan信息和当前的隐变量信息
                    sent_cond_z_embed = tf.concat((sent_cond_embed, sent_z), 1)

                    # 计算KL散度  KL_loss累加
                    with tf.name_scope("KL_divergence"):
                        # loss_mask:[batch,],如果某句子没有第i个seg，则不计此处KL散度
                        divergence = loss_mask * self.KL_divergence(sent_prior_mu, sent_prior_logvar,
                                                                    sent_post_mu, sent_post_logvar, False)
                        KL_loss += tf.reduce_sum(divergence) # / effective_cnt

                    # 计算类型的loss， 类型是当前句子第i个seg描述了什么类
                    with tf.name_scope("type_loss"):
                        if self.config.PHVM_use_type_info:
                            # 上一步plan信息，当前隐变量，当前句子当前seg的kv信息
                            # type_input:[batch, latent_decoder_dim+sent_latent_dim+encoder_dim] 
                            type_input = tf.concat((sent_cond_z_embed, gbow), 1)
                            #               [batch, latent_decoder_dim+sent_latent_dim+encoder_dim]->   type_fc_1
                            #               [batch, type_dim]->                                         type_fc_2
                            # type_logit:   [batch, type_vocab_size]
                            type_logit = type_fc_2(tf.tanh(type_fc_1(type_input)))
                            # 这里的type_logit为什么用普通的扩展就可作为结果？
                            # 答：type的含义是当前句子第i个seg描述了什么类，做比较的label是
                            # sent_type[batch,seg_i_type_num]记录了当前句子第i个seg都有什么类，该属性并无时序性
                            # 因此，普通的扩展得到的即对于当前句子第i个seg都有什么类的结果，不需要针对seg的第一个词或是第二个词有差别的处理
                            # type_logit:   [batch, seg_i_type_num, type_vocab_size]
                            type_logit = tf.tile(tf.expand_dims(type_logit, 1), [1, tf.shape(sent_type)[1], 1])
                            type_crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sent_type,    #[batch, seg_i_type_num]
                                                                                           logits=type_logit)   #[batch, seg_i_type_num, type_vocab_size]
                            type_crossent = loss_mask * tf.reduce_sum(type_crossent * sent_type_mask, 1)
                            type_loss += tf.reduce_sum(type_crossent) # / effective_cnt
                        else:
                            type_loss = 0.0

                    # 计算生成句子的loss， 为当前句子第i个seg的句子
                    with tf.name_scope("sent_deocde"):
                        '''
                            此处应该是利用     上一步的plan信息，当前隐变量信息，当前句子当前seg的kv信息和type信息作为输入
                            得到sent_dec_state  为随机生成的神经网络的初始状态
                        '''
                        with tf.variable_scope("sent_dec_state"):
                            # sent_dec_input:包含上一步的plan信息，当前隐变量信息，当前句子当前seg的kv信息和type信息
                            # sent_dec_input:[batch, latent_decoder_dim+encoder_dim(+type_dim)]
                            if self.config.PHVM_use_type_info:
                                sent_dec_input = tf.concat((sent_cond_z_embed, gbow, sent_type_embed), 1)
                            else:
                                sent_dec_input = tf.concat((sent_cond_z_embed, gbow), 1)
                            sent_dec_state = []
                            for _ in range(self.config.PHVM_decoder_num_layer):
                                # tmp:[batch, decoder_dim]   其实用了默认均匀随机分布得到结果，其中只是随机进行了纬度转换
                                tmp = tf.layers.dense(sent_dec_input, self.config.PHVM_decoder_dim)
                                if self.config.PHVM_rnn_type == 'lstm':
                                    tmp = tf.nn.rnn_cell.LSTMStateTuple(c=tmp, h=tmp)
                                sent_dec_state.append(tmp)
                            # sent_dec_state:[(decoder_num_layer,)batch,decoder_dim]
                            if self.config.PHVM_decoder_num_layer > 1:
                                sent_dec_state = tuple(sent_dec_state)
                            else:
                                sent_dec_state = sent_dec_state[0]

                        '''
                            sent_group:RNNencoder的输出[batch, seg_kv_num, encoder_dim*2] 指定句子的第i个seg中的kv的encoder形式
                            safe_sent_group_len:mask用，避免多余的空kv编码影响attention运算
                        '''
                        with tf.variable_scope("attention"):
                            attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.config.PHVM_decoder_dim,   # 注意力权重的size
                                                                                    sent_group,                     
                                                                                    memory_sequence_length=safe_sent_group_len) # [batch,] 表示需要mask的长度
                        # 为decoder封装一层attention
                        # decoder:decoder_num_layer层的decoder_dim循环神经网络 
                        # attention_mechanism:输入为sent_group[batch, seg_kv_num, encoder_dim*2]的attention机制，
                        # 输出为[batch, seg_kv_num, decoder_dim]
                        # train_decoder:[batch, seg_kv_num, decoder_dim]
                        train_decoder = tf.contrib.seq2seq.AttentionWrapper(decoder, attention_mechanism,
                                                                    attention_layer_size=self.config.PHVM_decoder_dim)
                        # 定下batch_size,并以刚刚基于sent_dec_input随机生成的参数初始化train_decoder的参数
                        train_encoder_state = train_decoder.zero_state(self.batch_size, dtype=tf.float32).clone(
                            cell_state=sent_dec_state)
                        # sent_input:[batch, seg_word_num, word_dim]    sent_lens: [batch,]
                        # 构建序列化输入，sent_lens用于mask 
                        helper = tf.contrib.seq2seq.TrainingHelper(sent_input, sent_lens, time_major=False)
                        # train_decoder:包含attention的编码层   helper:序列化输入   
                        # train_encoder_state:初始化的网络参数  output_layer:dense tgt_vocab_size
                        basic_decoder = tf.contrib.seq2seq.BasicDecoder(train_decoder, helper, train_encoder_state,
                                                                        output_layer=projection)
                        # fout:(rnn_outpues, sampleid)
                        # rnn_outputs:[batch, seg_word_num, vocab_size]
                        # sampleid:[batch_size,]    最终的编码结果
                         # fstate:   [batch, seg_word_num, tgt_vocab_size]   ？？？
                        with tf.variable_scope("dynamic_decoding"):
                            fout, fstate, flens = tf.contrib.seq2seq.dynamic_decode(basic_decoder, impute_finished=True)
                        # sent_logit:[batch, seg_word_num, vocab_size]
                        sent_logit = fout.rnn_output
                        # cali_sent_output:[batch, seg_word_num]    截取多于的seg_word_num为空
                        cali_sent_output = sent_output[:, :tf.shape(sent_logit)[1]]
                        # 计算根据attention的decoder输出的词与实际词的交叉熵
                        # cali_sent_output:[batch, seg_word_num] 实际值
                        # sent_logit:decoder输出的值 [batch, seg_word_num, vocab_size]
                        # sent_crossent:[batch, seg_word_num]   item的int不超过vocab_size
                        sent_crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=cali_sent_output,
                                                                                       logits=sent_logit)
                        # cali_sent_mask: [batch, seg_word_num]
                        cali_sent_mask = sent_mask[:, :tf.shape(sent_logit)[1]]
                        sent_crossent = loss_mask * tf.reduce_sum(sent_crossent * cali_sent_mask, axis=1)
                        sent_rec_loss += tf.reduce_sum(sent_crossent) # / effective_cnt

                        # 计算当前生成句子第i个seg的句子向量损失
                        with tf.name_scope("bow_loss"):
                            #           [batch, latent_decoder_dim+encoder_dim(+type_dim)]  ->  bow_fc_1
                            #           [batch, bow_hidden_dim] ->                              bow_fc_2
                            # bow_logit:[batch, tgt_vocab_size]
                            bow_logit = bow_fc_2(tf.tanh(bow_fc_1(sent_dec_input)))
                            # bow_logit:[batch, seg_word_num, tgt_vocab_size]
                            bow_logit = tf.tile(tf.expand_dims(bow_logit, 1), [1, tf.shape(sent_output)[1], 1])
                            bow_crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sent_output,
                                                                                          logits=bow_logit)
                            bow_crossent = loss_mask * tf.reduce_sum(bow_crossent * sent_mask, axis=1)
                            bow_loss += tf.reduce_sum(bow_crossent) # / effective_cnt

                        # 取decoder最后一层的state当做下一个seg的sent_state
                        with tf.variable_scope("sent_state_update"):
                            sent_state = fstate.cell_state[self.config.PHVM_decoder_num_layer - 1]
                            # if self.config.PHVM_rnn_type == 'lstm':
                            #     sent_top_input = fstate.cell_state[self.config.PHVM_decoder_num_layer - 1].h
                            # else:
                            #     sent_top_input = fstate.cell_state[self.config.PHVM_decoder_num_layer - 1]
                            # _, sent_state = sent_top_encoder(sent_top_input, sent_state)

                    return i + 1, group_state, gbow, plan_state, sent_state, sent_z, \
                           stop_sign, sent_rec_loss, group_rec_loss, KL_loss, type_loss, bow_loss

                # group_state:[batch, group_decoder_dim]  dec_input包含有句子大类的信息，kv对的信息，以及句子本身的信息
                group_state = group_init_state_fc(dec_input)
                # gbow:[batch, encoder_dim(*2)]     init_gbow是随机生成的长encoder_dim(*2)的变量
                gbow = tf.tile(init_gbow, [self.batch_size, 1])
                # 输入是dec_input和group_embed的结合 
                # dec_input包含有句子大类的信息，kv对的信息，以及句子本身的信息
                # group_embed是该句子的句向量表达  kv所属哪个seg的信息
                # plan_state: [batch, latent_decoder_dim]  
                plan_state = plan_init_state_fc(tf.concat((dec_input, group_embed), 1))
                # zero_state将循环层的输入的batch_size定死
                # 取decoder的最后一层为sent_state   大小为[batch, decoder_dim]
                sent_state = decoder.zero_state(self.batch_size, dtype=tf.float32)[
                    self.config.PHVM_decoder_num_layer - 1]
                # sent_z:[batch, sent_latent_dim] 的0tensor
                sent_z = tf.zeros(shape=(self.batch_size, self.config.PHVM_sent_latent_dim), dtype=tf.float32)
                # stop_sign: [seg_num, None, ] ???
                stop_sign = tf.TensorArray(dtype=tf.float32, element_shape=(None,),
                                           size=tf.shape(self.input.group)[1])
                # 循环得到这些结果
                _, group_state, gbow, plan_state, sent_state, sent_z, \
                stop_sign, sent_rec_loss, group_rec_loss, KL_loss, type_loss, bow_loss = \
                    tf.while_loop(train_cond, train_body,
                                  loop_vars=(0, group_state, gbow, plan_state, sent_state, sent_z, stop_sign,
                                             0.0, 0.0, 0.0, 0.0, 0.0))

                # 整合训练的损失
                with tf.name_scope("loss_computation"):
                    # stop_logit:[batch, seg_num]     stack将整个array变成tensor  
                    stop_logit = tf.transpose(stop_sign.stack(), [1, 0])
                    # stop_label:[batch, seg_num]
                    stop_label = tf.one_hot(self.input.group_cnt - 1, tf.shape(stop_logit)[1])
                    stop_crossent = tf.nn.sigmoid_cross_entropy_with_logits(logits=stop_logit, labels=stop_label)
                    # stop_mask: [batch, seg_num]
                    stop_mask = tf.sequence_mask(self.input.group_cnt, tf.shape(stop_logit)[1], dtype=tf.float32)
                    # 判断终止时所测的loss
                    self.stop_loss = tf.reduce_mean(tf.reduce_sum(stop_crossent * stop_mask, 1))
                    # 整合所有loss， 
                    # sent_rec_loss为decoder的下一状态的loss
                    # group_rec_loss 为判断seg有哪些kv对的损失
                    # sent_KL_divergence为每个seg KL损失的累加 KL的两个分布均有当前得到的plan信息，当前句子当前seg的kv对信息，
                    #                    post还有当前句子当前seg的编码信息和当前句子的大类信息
                    # type_loss 计算类型的loss， 类型是当前句子第i个seg描述了什么类
                    # bow_loss 计算当前生成句子第i个seg的句子对应的词的损失，由dense得到，未经历decoder
                    self.sent_rec_loss = sent_rec_loss
                    self.group_rec_loss = group_rec_loss
                    self.sent_KL_divergence = KL_loss
                    sent_KL_weight = tf.minimum(1.0, tf.to_float(self.global_step) / self.config.PHVM_sent_full_KL_step)
                    self.type_loss = type_loss
                    self.bow_loss = bow_loss / tf.to_float(self.batch_size)
                    # 控制KL损失的大小，避免刚开始学习不到信息，随着时间增长，再使KL信息正常
                    anneal_sent_KL = sent_KL_weight * self.sent_KL_divergence
                    anneal_plan_KL = plan_KL_weight * self.plan_KL_divergence
                    # 把多个损失相加，为误差下界
                    self.elbo_loss = self.sent_rec_loss + self.group_rec_loss + self.type_loss + \
                                     self.sent_KL_divergence + self.plan_KL_divergence
                    self.elbo_loss /= tf.to_float(self.batch_size)
                    # 带权重的KL损失，这里两种损失意义一样，只不过这个考虑了刚开始无法学习到知识的情况
                    self.anneal_elbo_loss = self.sent_rec_loss + self.group_rec_loss + self.type_loss + \
                                            anneal_sent_KL + anneal_plan_KL
                    self.anneal_elbo_loss /= tf.to_float(self.batch_size)
                    # 训练损失再加上停止损失和直接的seg词的损失，相当于一个残差
                    self.train_loss = self.anneal_elbo_loss + self.stop_loss + self.bow_loss

                # 梯度更新参数
                with tf.name_scope("update"):
                    params = tf.trainable_variables()                   # 需要训练的变量
                    gradients = tf.gradients(self.train_loss, params)   # 得到梯度
                    clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, 5) # 用于解决梯度爆炸

                    self.learning_rate = self.get_learning_rate()               #学习率
                    self.optimizer = tf.train.AdamOptimizer(self.learning_rate) #优化器
                    self.update = self.optimizer.apply_gradients(zip(clipped_gradients, params),
                                                                 global_step=self.global_step) # 记录迭代多少次

                # 统计参数显示
                with tf.name_scope("summary"):
                    self.gradient_summary = [tf.summary.scalar("gradient_norm", gradient_norm),
                                             tf.summary.scalar("clipped gradient_norm", tf.global_norm(clipped_gradients))]
                    self.train_summary = tf.summary.merge([tf.summary.scalar("learning rate", self.learning_rate),
                                                           tf.summary.scalar("train_loss", self.train_loss),
                                                           tf.summary.scalar("elbo", self.elbo_loss),
                                                           tf.summary.scalar("sent_KL_divergence", self.sent_KL_divergence / tf.to_float(self.batch_size)),
                                                           tf.summary.scalar("anneal_sent_KL", anneal_sent_KL / tf.to_float(self.batch_size)),
                                                           tf.summary.scalar("anneal_plan_KL", anneal_plan_KL / tf.to_float(self.batch_size)),
                                                           tf.summary.scalar("plan_KL_divergence", self.plan_KL_divergence / tf.to_float(self.batch_size)),
                                                           tf.summary.scalar("sent_rec_loss", self.sent_rec_loss / tf.to_float(self.batch_size)),
                                                           tf.summary.scalar("group_rec_loss", self.group_rec_loss / tf.to_float(self.batch_size)),
                                                           tf.summary.scalar("type_loss", self.type_loss / tf.to_float(self.batch_size)),
                                                           tf.summary.scalar("stop_loss", self.stop_loss),
                                                           tf.summary.scalar("bow_loss", self.bow_loss)]
                                                          + self.gradient_summary)
            # 推断
            with tf.name_scope("infer"):
                with tf.name_scope("group"):
                    with tf.name_scope("group_decode"):
                        '''
                            if i == 0:True
                            else:
                                if min(stop) == 0:
                                    if i >= max_sent_cnt:
                                        False
                                    else:
                                        True
                                else:
                                    False
                        '''
                        def group_cond(i, group_state, gbow, groups, glens, stop):
                            return tf.cond(tf.equal(i, 0),                                  # 是不是第1个
                                           lambda: True,                                    # 第一个必定继续进行
                                           lambda: tf.cond(tf.equal(tf.reduce_min(stop), 0),# 判断stop中的最小值是否为0
                                                           lambda: tf.cond(
                                                               tf.greater_equal(i, self.config.PHVM_max_sent_cnt),  # 判断i>=max_sent_cnt
                                                               lambda: False,               # 是为假
                                                               lambda: True),               # 否为真
                                                           lambda: False))                  # 不为0时为假
                        
                        def group_body(i, group_state, gbow, groups, glens, stop):
                            '''
                            --parames:
                                i:第i个seg
                                group_state:[batch, group_decoder_dim]上一个group_decoder输出的状态
                                gbow:       [batch, encoder_dim*2] 上一次判断对应seg有哪些kv对的编码
                                groups:     [batch, seg_num, kv_num]  记录下每一个句子的对应seg都有哪些kv对
                                glens:      [batch, seg_num]        记录下每一个seg都有多少kv对
                                stop:       [batch, ]               记录batch在第几个seg处停止
                            '''
                            with tf.variable_scope("decode_group", reuse=True):
                                # 通过编码输入gbow和decoder前一状态group_state得到新的输出和状态
                                # gout:[batch, group_decoder_dim]
                                # group:state
                                gout, group_state = group_decoder(gbow, group_state)
                            #               [batch, group_decoder_dim]              ->stop_clf
                            #               [batch, 1]                              ->squeeze
                            # next_stop:    [batch,]            与阈值的比较，判断根据group的编码得到的  true/false
                            next_stop = tf.greater(tf.sigmoid(tf.squeeze(stop_clf(gout), axis=1)),
                                                   self.config.PHVM_stop_threshold)
                            # stop:[batch,] 值为i+1  stop为假，next_stop为真的时候才记录，表现了下一步停止的意思
                            stop += tf.cast(tf.equal(stop, 0), dtype=tf.int32) * tf.cast(next_stop, dtype=tf.int32) * (
                                        i + 1)

                            # tile_gout:[batch, kv_num, group_decoder_dim]
                            tile_gout = tf.tile(tf.expand_dims(gout, 1), [1, tf.shape(src_encoder_output)[1], 1])
                            # group_fc_input:[batch, kv_num, encoder_dim+group_decoder_dim]  包含句子的kv对信息，和group的信息
                            group_fc_input = tf.concat((src_encoder_output, tile_gout), 2)
                            #               [batch, kv_num, encoder_dim+group_decoder_dim]      ->
                            #               [batch, kv_num, encoder_dim]                        ->
                            # group_logit:  [batch, kv_num] 包含整个句子kv对的信息和当前group_decoder的输出的信息 含义是当前seg出现对应kv对的概率
                            group_logit = tf.squeeze(group_fc_2(tf.tanh(group_fc_1(group_fc_input))), 2)

                            # 获取指定句子获取第i个seg的kv对都有哪些， 根据group_decoder决定
                            def select(group_prob, max_gcnt):
                                '''
                                输入:
                                    group_prob:[batch, kv_num]  该seg出现指定kv的概率
                                    max_gcnt:   int:kv_num
                                输出:
                                    gid:[batch, kv_num] item(batch,kv_id)   指定句子第i个seg描述的kv对，最多kv_num个
                                    glen:[batch,]
                                '''
                                gid = []
                                glen = []
                                for gfid, prob in enumerate(group_prob):
                                    # gfid: batch编号
                                    tmp = []
                                    max_gsid = -1
                                    max_p = -1
                                    for gsid, p in enumerate(prob):
                                        # gsid:kv编号
                                        # p:对应kv的概率值
                                        if p >= self.config.PHVM_group_selection_threshold:
                                            tmp.append([gfid, gsid])    # 如果概率超过阈值 则记录对应batch和kv编号
                                        if p > max_p:                   # 记录最大的概率
                                            max_gsid = gsid
                                            max_p = p
                                    if len(tmp) == 0:                   # 如果没有超过阈值的，选择最大的添加
                                        tmp.append([gfid, max_gsid])    
                                    # tmp:[seg_kv_num, ]    item:[batch_id, kv_id]
                                    gid.append(tmp)                     # 记录此seg所要描述的kv对   
                                    glen.append(len(tmp))               # int:当前seg有几个kv对
                                # gid:[batch, seg_kv_num]
                                # glen:[batch,]
                                for item in gid:
                                    if len(item) < max_gcnt:
                                        item += [[0, 0]] * (max_gcnt - len(item))   #填充gid的seg_kv_num到kv_num
                                # [batch, kv_num] item(batch_id, kv_id)
                                return np.array(gid, dtype=np.int32), np.array(glen, dtype=np.int32) 
                            
                            # src_mask:[batch, kv_num]
                            src_mask = tf.sequence_mask(self.input.input_lens, tf.shape(group_logit)[1],
                                                        dtype=tf.float32)
                            # group:[batch, kv_num]
                            group_prob = tf.sigmoid(group_logit) * src_mask

                            # gid:[batch, kv_num] item(batch,kv_id)
                            # glen:[batch,]
                            gid, glen = tf.py_func(select, [group_prob, tf.shape(src_encoder_output)[1]],
                                                   [tf.int32, tf.int32])
                            # gid:[batch, kv_num, 2]
                            gid = tf.reshape(gid, (self.batch_size, -1, 2))
                            # glen:[batch,]
                            glen = tf.reshape(glen, (-1,))
                            # expanded_glen:[batch,1]
                            expanded_glen = tf.expand_dims(glen, 1)
                            # groups:[batch, seg_num, kv_num] 针对每句话，不断添加需要描述的kv对，添加在axis1，每个seg加1
                            groups = tf.concat((groups, tf.transpose(gid[:, :, 1:], [0, 2, 1])), 1)
                            # glens:[batch, seg]    每个seg加1，最后得到每句话每个seg的
                            glens = tf.concat((glens, expanded_glen), 1)
                            # group:[batch, kv_num, encoder_dim*2]  gid选取id
                            group = tf.gather_nd(src_encoder_output, gid)
                            # expanded_group_mask:[batch, kv_num, 1] 利用对应seg的kv数量mask掉多于信息
                            group_mask = tf.sequence_mask(glen, tf.shape(group)[1], dtype=tf.float32)
                            expanded_group_mask = tf.expand_dims(group_mask, 2)
                            # gbow:[batch, encoder_dim]     把指定seg对应的kv编码求均值得到对应的seg的kv向量
                            gbow = tf.reduce_sum(group * expanded_group_mask, axis=1) / tf.to_float(
                                expanded_glen)

                            return i + 1, group_state, gbow, groups, glens, stop

                        # 获取group_state的变量形式, 统计
                        if self.config.PHVM_rnn_type == 'lstm':
                            group_state_shape = tf.nn.rnn_cell.LSTMStateTuple(c=tf.TensorShape([None, None]),
                                                                              h=tf.TensorShape([None, None]))
                        else:
                            group_state_shape = tf.TensorShape([None, None])

                        # 定下各张量的形状
                        shape_invariants = (tf.TensorShape([]), # i
                                            group_state_shape, # group_state
                                            tf.TensorShape([None, None]), # gbow
                                            tf.TensorShape([None, None, None]), # groups
                                            tf.TensorShape([None, None]), # glens
                                            tf.TensorShape([None]) # stop
                        )
                        # group_state:[batch, group_decoder_dim]
                        group_state = group_init_state_fc(dec_input)
                        # gbow:[batch, encoder_dim(*2)]
                        gbow = tf.tile(init_gbow, [self.batch_size, 1])
                        # groups:[batch,1, kv_num]
                        groups = tf.zeros((self.batch_size, 1, tf.shape(src_encoder_output)[1]), dtype=tf.int32)
                        # glens:[batch,]
                        glens = tf.zeros((self.batch_size, 1), dtype=tf.int32)
                        # stop[batch,]
                        stop = tf.zeros((self.batch_size,), dtype=tf.int32)
                        '''
                                group_state:[batch, group_decoder_dim]上一个group_decoder输出的状态
                                gbow:       [batch, encoder_dim*2] 上一次判断对应seg有哪些kv对的编码
                                groups:     [batch, seg_num, kv_num]  记录下每一个句子的对应seg都有哪些kv对
                                glens:      [batch, seg_num]        记录下每一个seg都有多少kv对
                                stop:       [batch, ]               记录batch在第几个seg处停止
                        '''
                        _, group_state, gbow, groups, glens, stop = \
                            tf.while_loop(group_cond, group_body,
                                          loop_vars=(0, group_state, gbow, groups, glens, stop),
                                          shape_invariants=shape_invariants)

                        # 将第一个，即初始状态去掉，无信息
                        groups = groups[:, 1:, :]
                        glens = glens[:, 1:]

                    with tf.name_scope("group_encode"):
                        gidx, group_bow, group_mean_bow, group_embed = self.gather_group(src_encoder_output,
                                                                                         groups,
                                                                                         glens,
                                                                                         stop,
                                                                                         group_encoder)
                # return i < seg_num
                def infer_cond(i, plan_state, sent_state, sent_z, translations):
                    return i < tf.shape(groups)[1]

                # 
                def infer_body(i, plan_state, sent_state, sent_z, translations):
                    '''
                        i:              第i个seg
                        plan_state:     [batch, latent_decoder_dim]   由上一状态的传递状态和传递隐变量构成
                        sent_state:     [batch, decoder_dim]            传递状态  由指定句子指定seg的kv对编码形式解码得到decoder的下一状态
                        sent_z:         [batch, sent_latent_dim]    向下一层传递的隐含状态,有当前seg的编码形式和当前plan的输入得到
                        translations:   [batch, 1, maximum_iterations] 补全的结果，是beam_decoder
                    '''

                    with tf.name_scope("group"):
                        gbow = group_mean_bow[:, i, :]      # [batch, encoder_dim*2] 指定句子第i个seg的编码形式
                        sent_group = group_bow[:, i, :, :]  # [batch, seg_kv_num, encoder_dim*2] 指定句子指定seg的kv对的编码形式
                        sent_glen = glens[:, i]             # [batch,]      指定句子指定seg有多少kv对

                    # plan_input:[batch, decoder_dim+sent_latent_dim]   获取plan的输入由上一步decoder的解码和送入的隐变量得到
                    if self.config.PHVM_rnn_type == 'lstm':
                        plan_input = tf.concat((sent_state.h, sent_z), 1)
                    else:
                        plan_input = tf.concat((sent_state, sent_z), 1)
                    
                    # sent_cond_embed/plan_state:   [batch, decoder_dim]
                    sent_cond_embed, plan_state = latent_decoder(plan_input, plan_state)

                    with tf.name_scope("sent_prior_network"):
                        # sent_prior_input:[batch, decoder_dim+encoder_dim*2]
                        sent_prior_input = tf.concat((sent_cond_embed, gbow), 1)
                        sent_prior_fc = prior_fc_layer(sent_prior_input)
                        # sent_prior_mu/sent_prior_logvar:  [batch, sent_latent_dim]
                        sent_prior_mu, sent_prior_logvar = tf.split(sent_prior_fc, 2, axis=1)
                        sent_z = self.sample_gaussian((self.batch_size, self.config.PHVM_sent_latent_dim),
                                                      sent_prior_mu,
                                                      sent_prior_logvar)
                        # sent_z:[batch, sent_latent_dim]
                        sent_z = tf.reshape(sent_z, (self.batch_size, self.config.PHVM_sent_latent_dim))

                    # sent_cond_z_embed:    [batch, decoder_dim+sent_latent_dim]    当前输出和经历了重采样的隐变量结合
                    sent_cond_z_embed = tf.concat((sent_cond_embed, sent_z), 1)

                    with tf.name_scope("type"):
                        if self.config.PHVM_use_type_info:
                            sent_type_input = tf.concat((sent_cond_z_embed, gbow), 1)   # [batch, decoder_dim+encoder_dim*2]
                            sent_type_logit = type_fc_2(tf.tanh(type_fc_1(sent_type_input)))# [batch, type_vocab_size]
                            sent_type_prob = tf.nn.softmax(sent_type_logit, dim=1)      # [batch, ]
                            sent_type_embed = tf.matmul(sent_type_prob, self.type_embedding)    #[batch, type_dim]  转换得到的type为embedding格式

                    with tf.name_scope("sent_deocde"):
                        with tf.variable_scope("sent_dec_state", reuse=True):
                            # sent_dec_input：[batch, decoder_dim+sent_latent_dim+encoder_dim*2(+type_dim)]
                            if self.config.PHVM_use_type_info:
                                sent_dec_input = tf.concat((sent_cond_z_embed, gbow, sent_type_embed), 1)
                            else:
                                sent_dec_input = tf.concat((sent_cond_z_embed, gbow), 1)
                            sent_dec_state = []
                            for _ in range(self.config.PHVM_decoder_num_layer):
                                tmp = tf.layers.dense(sent_dec_input, self.config.PHVM_decoder_dim)
                                if self.config.PHVM_rnn_type == 'lstm':
                                    tmp = tf.nn.rnn_cell.LSTMStateTuple(c=tmp, h=tmp)
                                sent_dec_state.append(tmp)
                            if self.config.PHVM_decoder_num_layer > 1:
                                sent_dec_state = tuple(sent_dec_state)
                            else:
                                sent_dec_state = sent_dec_state[0]
                            # sent_dec_state:[batch, decoder_dim]
                        
                        # tile_glen:[batch*beam_width, ]    逐项复制，不是所有的都复制一次
                        tile_glen = tf.contrib.seq2seq.tile_batch(sent_glen, multiplier=self.config.PHVM_beam_width)
                        # tile_group:[batch*beam_width, seg_kv_num, encoder_dim*2]
                        tile_group = tf.contrib.seq2seq.tile_batch(sent_group, multiplier=self.config.PHVM_beam_width)
                        # tile_att: 输出为decoder_dim的attention，输入是[batch*beam_width, seg_kv_num, encoder_dim*2]
                        #           len为tile_glen
                        with tf.variable_scope("attention", reuse=True):
                            tile_att = tf.contrib.seq2seq.LuongAttention(self.config.PHVM_decoder_dim,
                                                                         tile_group,
                                                                         memory_sequence_length=tile_glen)
                        # 为decoder添加attention
                        infer_decoder = tf.contrib.seq2seq.AttentionWrapper(decoder, tile_att,
                                                                attention_layer_size=self.config.PHVM_decoder_dim)
                        # tile_encoder_state:[batch*beam_width, decoder_dim]
                        tile_encoder_state = tf.contrib.seq2seq.tile_batch(sent_dec_state,
                                                                           multiplier=self.config.PHVM_beam_width)
                        # 初始化状态
                        decoder_initial_state = infer_decoder.zero_state(
                            self.batch_size * self.config.PHVM_beam_width,
                            dtype=tf.float32).clone(cell_state=tile_encoder_state)
                        # 封装decoder成beam_decoder, 比贪心解码得到的句子更好
                        self.beam_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                            cell=infer_decoder,
                            embedding=self.word_embedding,
                            start_tokens=tf.tile([self.start_token], [self.batch_size]),
                            end_token=self.end_token,
                            initial_state=decoder_initial_state,
                            beam_width=self.config.PHVM_beam_width,
                            output_layer=projection,
                            length_penalty_weight=0.0
                        )
                        # 
                        with tf.variable_scope("dynamic_decoding", reuse=True):
                            # fout:     [batch, seg_kv_num, tgt_vocab_size]
                            # fstate:   [decoder_layer, batch*beam_width, deco]
                            fout, fstate, flen = tf.contrib.seq2seq.dynamic_decode(self.beam_decoder,
                                                        output_time_major=False,
                                                        maximum_iterations=self.config.PHVM_maximum_iterations)
                            # predicted_ids: [batch, seg_kv_num, beam_size]
                            # sent_output: [batch, seg_kv_num ] 选取最大可能的结论  值为vocab_size以下的整数
                            sent_output = tf.transpose(fout.predicted_ids, [0, 2, 1])[:, 0, :]
                            # 将sent_output小于0的值归0
                            sent_output = sent_output * tf.to_int32(tf.greater_equal(sent_output, 0))
                            # dist:maximum_iterations-seg_kv_num
                            dist = self.config.PHVM_maximum_iterations - tf.shape(sent_output)[1]
                            # padded_sent_out 把sent_out 补全到[batch, maximum_iterations]
                            padded_sent_output = tf.cond(tf.greater(dist, 0),   # 如果seg_kv_num没超过最大迭代次数，这里的seg_kv_num是由decoder决定的
                                                         lambda: tf.concat(
                                                             (sent_output,      
                                                              tf.zeros((self.batch_size, dist), dtype=tf.int32)), 1),
                                                         lambda: sent_output)
                            # translations:[batch, seg_num, maxmum_iterations]   一步步补全translations，每次有一个seg
                            translations = tf.concat((translations, tf.expand_dims(padded_sent_output, 1)), 1)
                            # pad_output:   [batch, seg_kv_num+1]  为原来生成的句子的前面添加1位
                            pad_output = tf.concat((tf.zeros((self.batch_size, 1), dtype=tf.int32), sent_output), 1)
                            # sent_lens:    [batch, ] 最大的长度
                            sent_lens = tf.argmax(tf.cast(tf.equal(pad_output, 1), dtype=tf.int32), 1,
                                                  output_type=tf.int32)
                            # sent_lens-1+  ==0时(seg_kv_num+1)
                            sent_lens = sent_lens - 1 + tf.to_int32(tf.equal(sent_lens, 0)) * (tf.shape(sent_output)[1] + 1)

                        # sent encode
                        with tf.variable_scope("attention", reuse=True):
                            att = tf.contrib.seq2seq.LuongAttention(self.config.PHVM_decoder_dim,
                                                                    sent_group,     #[batch, seg_kv_num, encoder_dim*2]
                                                                    memory_sequence_length=sent_glen)
                        
                        infer_encoder = tf.contrib.seq2seq.AttentionWrapper(decoder, att,
                                                            attention_layer_size=self.config.PHVM_decoder_dim)
                        # sent_output:[batch, seg_kv_num ]
                        # sent_input:[batch, seg_kv_num, word_dim]
                        sent_input = tf.nn.embedding_lookup(self.word_embedding, sent_output)
                        # encoder_state:赋值未sent_dec_state
                        encoder_state = infer_encoder.zero_state(self.batch_size, dtype=tf.float32).clone(
                            cell_state=sent_dec_state)
                        helper = tf.contrib.seq2seq.TrainingHelper(sent_input, sent_lens, time_major=False)
                        basic_decoder = tf.contrib.seq2seq.BasicDecoder(infer_encoder, helper, encoder_state,
                                                                        output_layer=projection)
                        with tf.variable_scope("dynamic_decoding", reuse=True):
                            # fout:[batch, seg_kv_num, decoder_dim]
                            # fstate:[decoder_num_layer, batch, decoder_dim]
                            fout, fstate, flens = tf.contrib.seq2seq.dynamic_decode(basic_decoder,
                                                                                    impute_finished=True)

                        with tf.variable_scope("sent_state_update", reuse=True):
                            sent_state = fstate.cell_state[self.config.PHVM_decoder_num_layer - 1]

                    return i + 1, plan_state, sent_state, sent_z, translations

                # 先为plan_state 开辟空间
                if self.config.PHVM_rnn_type == 'lstm':
                    plan_state_shape = tf.nn.rnn_cell.LSTMStateTuple(c=tf.TensorShape([None, None]),
                                                                     h=tf.TensorShape([None, None]))
                else:
                    plan_state_shape = tf.TensorShape([None, None])
                # 同sent_state_shape
                sent_state_shape = plan_state_shape
                # 开辟infer_body的动态tensor空间
                shape_invariants = (tf.TensorShape([]), # i
                                    plan_state_shape, # plan_state
                                    sent_state_shape, # sent_state
                                    tf.TensorShape([None, None]), # sent_z

                                    tf.TensorShape([None, None, None]), # translations
                                    )
                # dec_input:    [batch, cate_dim + encoder_dim(*2) + plan_latent_dim]   包含大类，kv编码，隐层变量
                # group_embed:  [batch, group_encoder_dim]                              指定句子中的句向量，源于seg中有无指定kv
                # plan_state:   [batch, latent_decoder_dim]             初始化plan_state
                plan_state = plan_init_state_fc(tf.concat((dec_input, group_embed), 1))
                # sent_state:   [batch_size, decoder_dim] decoder清空上一状态，把最后一层的输出作为sent_state 初始化一个状态
                sent_state = decoder.zero_state(self.batch_size, dtype=tf.float32)[
                    self.config.PHVM_decoder_num_layer - 1]
                # sent_z:       [batch, sent_latent_dim]
                sent_z = tf.zeros(shape=(self.batch_size, self.config.PHVM_sent_latent_dim), dtype=tf.float32)
                # translations: [batch, 1, maximum_iterations]
                translations = tf.zeros((self.batch_size, 1, self.config.PHVM_maximum_iterations), dtype=tf.int32)

                _, plan_state, sent_state, sent_z, translations = \
                    tf.while_loop(infer_cond,
                                  infer_body,
                                  loop_vars=(0, plan_state, sent_state, sent_z, translations),
                                  shape_invariants=shape_invariants)
                # 赋值 stop:记录batch在第几个seg处停止
                # groups [batch, seg_num, kv_num]  记录下每一个句子的对应seg都有哪些kv对
                # glens:[batch, seg_num]        记录下每一个seg都有多少kv对
                # translations:[batch, seg_num, maximum_iterations]
                self.stop = stop + tf.cast(tf.equal(stop, 0), dtype=tf.int32) * self.config.PHVM_max_sent_cnt
                self.groups = groups
                self.glens = glens
                self.translations = translations[:, 1:, :]

    # 
    def get_global_step(self):
        return self.sess.run(self.global_step)  # 返回图中的global_step

    def train(self, batch_input):
        # feed_dict是把未确定的tensor赋值
        feed_dict = {key: val for key, val in zip(self.input, batch_input)}
        feed_dict[self.keep_prob] = 1 - self.config.PHVM_dropout
        feed_dict[self.train_flag] = True
        # update: 获取梯度   
        # global_step:全局步数，迭代了多少次
        # train_loss:训练的损失
        # train_summary:总结
        _, global_step, train_loss, summary = self.sess.run(
            (self.update, self.global_step, self.train_loss, self.train_summary), feed_dict=feed_dict)
        return global_step, train_loss, summary

    def eval(self, batch_input):
        # 评估
        feed_dict = {key: val for key, val in zip(self.input, batch_input)}
        feed_dict[self.keep_prob] = 1
        feed_dict[self.train_flag] = True
        global_step, loss = self.sess.run((self.global_step, self.elbo_loss), feed_dict=feed_dict)
        return global_step, loss

    def infer(self, batch_input):
        # 推断
        feed_dict = {self.input.key_input: batch_input.key_input,
                     self.input.val_input: batch_input.val_input,
                     self.input.input_lens: batch_input.input_lens,
                     self.input.category: batch_input.category,
                     self.input.text: np.array([[0]] * len(batch_input.category)),
                     self.input.slens: np.array([1] * len(batch_input.category)),
                     }
        feed_dict[self.keep_prob] = 1
        feed_dict[self.train_flag] = False
        # translations:[batch, seg_num, maximum_iterations] batch的对应seg的句子
        stop, groups, glens, translations = self.sess.run((self.stop, self.groups, self.glens, self.translations), feed_dict=feed_dict)
        return self._agg_group(stop, translations)

    def _agg_group(self, stop, text):
        translation = []
        # gcnt, 各batch在第几个seg处停止
        # sent, 各batch的各seg的句子
        for gcnt, sent in zip(stop, text):
            # 截取有效句子
            sent = sent[:gcnt, :]
            desc = []
            # seg每个seg的句子
            for segId, seg in enumerate(sent):
                for wid in seg:
                    if wid == self.end_token:
                        break
                    elif wid == self.start_token:
                        continue
                    else:
                        desc.append(wid)  # desc 将该seg的id记下 
            translation.append(desc)    # 将逐步将每个batch的各个seg合在一起形成一句话
        #[batch, sen_num]
        max_len = 0
        # 记录最大长度
        for sent in translation:
            max_len = max(max_len, len(sent))
        # 为每个batch加上终止符做pad
        for i, sent in enumerate(translation):
            translation[i] = [sent + [self.end_token] * (max_len - len(sent))]
        return translation
