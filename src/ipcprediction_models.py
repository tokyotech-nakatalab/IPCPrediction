import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
      
    pos_encoding = angle_rads[np.newaxis, ...]
      
    return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def scaled_dot_product_attention(q, k, v, mask):

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    if mask is not None:
      scaled_attention_logits += (mask * -1e9)  

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
          
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
      
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
            
        return output, attention_weights



def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
      
    def call(self, x, mask=None):

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        
        return out2

class MHA_collector(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MHA_collector, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
            
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
        
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
            
        return output, attention_weights


class Collect_outputs(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate):
        super(Collect_outputs,self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.collector = self.add_weight(name="collector" ,shape=[1,1,d_model])

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, mask):

        batch_size = tf.shape(x)[0]
        keys = tf.tile(self.collector, [batch_size,1,1])

        attn_output, _ = self.mha(x,x,keys, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(keys + attn_output)  # (batch_size, input_seq_len, d_model)
        
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        
        return out2


class Encoder_TRANS(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                maximum_position_encoding, rate=0.1):
        super(Encoder_TRANS, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                                self.d_model)
        
        
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                          for _ in range(num_layers)]
      
        self.dropout = tf.keras.layers.Dropout(rate)
          
    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]
        
        # 埋め込みと位置エンコーディングを合算する
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
          x = self.enc_layers[i](x, training, mask)
        
        return x  # (batch_size, input_seq_len, d_model)


class Encoder_SET_TRANS(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                maximum_position_encoding, rate=0.1):
        super(Encoder_SET_TRANS, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model,mask_zero=True)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                          for _ in range(num_layers)]
      
        self.dropout = tf.keras.layers.Dropout(rate)
          
    def call(self, x, training, mask):

        
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
          x = self.enc_layers[i](x, training, mask)
        
        return x  # (batch_size, input_seq_len, d_model)

class Encoder_RATIO_TRANS(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, embedding_matrix,
                maximum_position_encoding, rate=0.1):
        super(Encoder_RATIO_TRANS, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model,mask_zero=True, weights = [embedding_matrix])
        
        
        self.enc_layers = [EncoderLayer(dff, num_heads, dff, rate) 
                          for _ in range(num_layers)]
      
        self.dropout = tf.keras.layers.Dropout(rate)#,noise_shape=(batch_size, words,1))#単語を丸ごと消すようなdropout
        self.get_words = Lambda(lambda x :x[:,0],mask=0)
        self.get_ratio = Lambda(lambda x :x[:,1],mask=0)
        self.mask0 = tf.keras.layers.Masking(mask_value=0)
        self.expand = Lambda(lambda x: K.expand_dims(x,axis=2))

          
    def call(self, x):

        words = x[:,0]
        ratio = x[:,1]
        ratio = self.expand(self.mask0(ratio))
        em = self.embedding(words)
        x = ratio * em
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.dropout(x)
        
        for i in range(self.num_layers):
          x = self.enc_layers[i](x)
        
        return x  # (batch_size, input_seq_len, dff)

class HMLN(tf.keras.layers.Layer):
    def __init__(self, NUM_CLS, NUM_SUBCLS, NUM_MAINGRP, NUM_SUBGRP, NUM_FI, dim_h ,rate):
        super(HMLN,self).__init__()
        self.cls1 = tf.keras.layers.Dense(dim_h,activation='tanh')
        self.cls2 = tf.keras.layers.Dense(dim_h,activation='tanh')
        self.subcls1 = tf.keras.layers.Dense(dim_h,activation='tanh')
        self.subcls2 = tf.keras.layers.Dense(dim_h,activation='tanh')
        self.maingrp1 = tf.keras.layers.Dense(dim_h,activation='tanh')
        self.maingrp2 = tf.keras.layers.Dense(dim_h,activation='tanh')
        self.subgrp1 = tf.keras.layers.Dense(dim_h,activation='tanh')
        self.subgrp2 = tf.keras.layers.Dense(dim_h,activation='tanh')
        self.fi1 = tf.keras.layers.Dense(dim_h,activation='tanh')
        self.fi2 = tf.keras.layers.Dense(dim_h,activation='tanh')
        self.cls3 = tf.keras.layers.Dense(dim_h,activation='sigmoid')
        self.cls4 = tf.keras.layers.Dense(NUM_CLS,activation='sigmoid')
        self.subcls3 = tf.keras.layers.Dense(dim_h,activation='sigmoid')
        self.subcls4 = tf.keras.layers.Dense(NUM_SUBCLS,activation='sigmoid')
        self.maingrp3 = tf.keras.layers.Dense(dim_h,activation='sigmoid')
        self.maingrp4 = tf.keras.layers.Dense(NUM_MAINGRP,activation='sigmoid')
        self.subgrp3 = tf.keras.layers.Dense(dim_h,activation='sigmoid')
        self.subgrp4 = tf.keras.layers.Dense(NUM_SUBGRP,activation='sigmoid')
        self.fi3 = tf.keras.layers.Dense(dim_h,activation='sigmoid')
        self.fi4 = tf.keras.layers.Dense(NUM_FI,activation='sigmoid')
        self.get1 = Lambda(lambda x: x[:,0,:])
        self.attn = tf.keras.layers.Attention()
        self.dropout = tf.keras.layers.Dropout(rate)
    
    def call(self,h_all,h_final,training):
        h_cls1 = self.cls1(h_final) 
        h_cls2 = self.cls2(h_cls1)
        h_subcls1 = self.subcls1(tf.concat([h_cls1,h_final],axis=1))
        h_subcls2 = self.subcls2(h_subcls1)
        h_maingrp1 = self.maingrp1(tf.concat([h_subcls1,h_final],axis=1))
        h_maingrp2 = self.maingrp2(h_maingrp1)
        h_subgrp1 = self.subgrp1(tf.concat([h_maingrp1,h_final],axis=1))
        h_subgrp2 = self.subgrp2(h_subgrp1)
        h_fi1 = self.subgrp1(tf.concat([h_subgrp1,h_final],axis=1))
        h_fi2 = self.subgrp2(h_fi1)
        c_cls = self.get1(self.attn([h_cls2,h_all])) 
        c_subcls = self.get1(self.attn([h_subcls2,h_all]))
        c_maingrp = self.get1(self.attn([h_maingrp2,h_all]))
        c_subgrp = self.get1(self.attn([h_subgrp2,h_all]))
        c_fi = self.get1(self.attn([h_fi2,h_all]))
        h_cls3 = self.cls3(tf.concat([h_cls2,c_cls],axis=1))
        h_subcls3 = self.subcls3(tf.concat([h_subcls2,c_subcls],axis=1))
        h_maingrp3 = self.maingrp3(tf.concat([h_maingrp2,c_maingrp],axis=1))
        h_subgrp3 = self.subgrp3(tf.concat([h_subgrp2,c_subgrp],axis=1))
        h_fi3 = self.fi3(tf.concat([h_fi2,c_fi],axis=1))
        p_cls = self.cls4(self.dropout(h_cls3,training=training))
        p_subcls = self.subcls4(self.dropout(tf.concat([h_subcls3,p_cls],axis=1),training=training))
        p_maingrp = self.maingrp4(self.dropout(tf.concat([h_maingrp3,p_subcls],axis=1),training=training))
        p_subgrp = self.subgrp4(self.dropout(tf.concat([h_subgrp3,p_maingrp],axis=1),training=training))
        p_fi = self.fi4(self.dropout(tf.concat([h_fi3,p_subgrp],axis=1),training=training))
        return p_cls,p_subcls,p_maingrp,p_subgrp,p_fi

class HSMLN(tf.keras.layers.Layer):
    def __init__(self, HIERALCHY,dim_h,num_h,rate=0.5,l2_norm=1e-5):
        super(HSMLN,self).__init__()
        self.N = HIERALCHY[0][-1][-1]
        self.alpha = (np.log(self.N))/(dim_h-1)
        self.H = len(HIERALCHY)
        self.label1 = [[tf.keras.layers.Dense(self.determine_dim(HIERALCHY[h][i][1]),activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(l2_norm)) for i in range(1,len(HIERALCHY[h]))] for h in range(self.H)]
        self.label3 = [[tf.keras.layers.Dense(HIERALCHY[h][i+1][0]-HIERALCHY[h][i][0],activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l2_norm)) for i in range(len(HIERALCHY[h])-1)] for h in range(self.H)]
        self.labelc = [tf.keras.layers.Dense(dim_h, kernel_regularizer=tf.keras.regularizers.l2(l2_norm)) for h in range(1,self.H)]
        self.mhas = [MultiHeadAttention(dim_h, num_h) for h in range(1,self.H)]
        self.dense = [tf.keras.layers.Dense(dim_h,activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(l2_norm)) for h in range(1,self.H)]
        self.HIERALCHY = HIERALCHY
        self.dropout = tf.keras.layers.Dropout(rate)
        self.get1 = Lambda(lambda x: x[:,0,:])
        self.attn = tf.keras.layers.Attention()

    def determine_dim(self,n):
        if n == self.N:
          return self.N
        elif n==1:
          return 1
        else:
          return int(1+np.ceil(np.log(n)/self.alpha))
      

    def one_step(self,input,attn,layerl,layer3):
        out = layerl(input)
        p = layer3(self.dropout(tf.concat([out,attn],axis=1)))
        return p,out

    def one_step_1st(self,input,layerl,layer3):
        out = layerl(input)
        p = layer3(self.dropout(out))
        return p,out


    def call(self,h_all,h_final):
        p_label = [[] for h in range(self.H)]
        h_label = [[] for h in range(self.H)]
        #classにおける処理
        p_tmp, h_tmp = self.one_step_1st(h_final,self.label1[0][0],self.label3[0][0])
        p_label[0] = p_tmp
        h_label[0].append(h_tmp)
        vec_tmp = self.labelc[0](p_label[0])
        a_tmp = self.get1(self.mhas[0](h_all,h_all,vec_tmp,None)[0])

        for h in range(1,self.H):#subclass以降
          hier_pa = self.HIERALCHY[h-1]
          for i_pa in range(len(hier_pa)-1):
            for i_ch in range(hier_pa[i_pa][0], hier_pa[i_pa+1][0]):
              input_local = h_label[h-1][i_pa]
              p_tmp,h_tmp = self.one_step(input_local, a_tmp, self.label1[h][i_ch], self.label3[h][i_ch])
              p_label[h].append(p_tmp)
              h_label[h].append(h_tmp)
          p_label[h] = tf.concat(p_label[h], axis=1)
          if not h == self.H -1:
            vec_tmp = self.labelc[h](p_label[h])
            a_tmp = self.get1(self.mhas[h](h_all,h_all,vec_tmp,None)[0])

        return p_label

class Decoder_Base(tf.keras.layers.Layer):
    def __init__(self,  NUM_FI):
        super(Decoder_Base,self).__init__()
        self.N = NUM_FI
        self.D1 = tf.keras.layers.Dense(256,activation='relu')
        self.D2 = tf.keras.layers.Dense(256,activation='relu')
        self.D3 = tf.keras.layers.Dense(NUM_FI,activation='sigmoid')

    def call(self,h_final,training):
        h_1 = self.D1(h_final)
        h_2 = self.D2(h_1)
        p = self.D3(h_2)

        return p

class Decoder_HARNN(tf.keras.layers.Layer):
    def __init__(self, HIERALCHY, fc_hidden_size, rate, alpha=0.5, l2_norm=1e-5):
        super(Decoder_HARNN, self).__init__()
        NUM_LABELS = 0
        for i in range(len(HIERALCHY)):
            NUM_LABELS += HIERALCHY[i][-1][0]
        self.dense_at1 = [tf.keras.layers.Dense(HIERALCHY[h][-1][0], activation='tanh', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(l2_norm)) for h in range(len(HIERALCHY))]
        self.dense_at2 = [tf.keras.layers.Dense(HIERALCHY[h][-1][0], use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(l2_norm)) for h in range(len(HIERALCHY))]
        self.dense_fc = [tf.keras.layers.Dense(fc_hidden_size, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_norm)) for h in range(len(HIERALCHY))]
        self.dense_lc = [tf.keras.layers.Dense(HIERALCHY[h][-1][0], activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l2_norm)) for h in range(len(HIERALCHY))]
        self.fc_gl = tf.keras.layers.Dense(fc_hidden_size, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_norm))
        self.higher_relu = tf.keras.layers.Dense(fc_hidden_size, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_norm))
        self.higher_sigm = tf.keras.layers.Dense(fc_hidden_size, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l2_norm))
        self.last_sigm = tf.keras.layers.Dense(NUM_LABELS, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l2_norm))
        self.HIERALCHY = HIERALCHY
        self.alpha = alpha
        self.dropout = tf.keras.layers.Dropout(rate)

    def _text_attention(self, input_x, h):
        weight = tf.keras.activations.softmax(self.dense_at2[h](self.dense_at1[h](input_x)),axis=-1)
        #out = tf.math.reduce_sum(tf.matmul(weight, input_x),axis=1)
        out = tf.einsum('ijk,ijl->il',weight, input_x)
        return weight, out
    
    def _local_layer(self, input_x, att_weight, h):
        """
        input_x : (batch_size, dim_h)
        att_weight : (batch_size, seq_length, num_cls)
        """
        score = self.dense_lc[h](input_x) #(batch_size, num_cls)
        visual = tf.multiply(att_weight, tf.expand_dims(score, 1)) #(batch_size, seq_length, num_cls)
        visual = tf.nn.softmax(visual)
        visual = tf.reduce_mean(visual, axis=-1) #(batch_size, seq_length)

        return score, visual
    
    def _highway_layer(self, input_x):
        out = self.higher_relu(input_x)
        t = self.higher_sigm(input_x)
        out = t * out + (1.-t)* input_x
        
        return out
            
    def call(self,lstm_out,lstm_pool):
        att_weight, att_out = self._text_attention(lstm_out, 0)
        local_input = tf.concat([lstm_pool, att_out], axis=-1)
        local_fc_out = self.dense_fc[0](local_input)
        fc_outs = [local_fc_out]
        scores, visual = self._local_layer(local_fc_out, att_weight, 0)
        scores_list = [scores]
        for h in range(1, len(self.HIERALCHY)):
            att_input = tf.multiply(lstm_out, tf.expand_dims(visual, -1))#ここが(307, 256)*(seq_length,1)になってる
            att_weight, att_out = self._text_attention(att_input, h)
            local_fc_out = self.dense_fc[h](local_input)
            fc_outs.append(local_fc_out)
            scores, visual = self._local_layer(local_fc_out, att_weight, h)
            scores_list.append(scores)
        fc_outs = self.fc_gl(tf.concat(fc_outs, axis=1))
        h_global = self.dropout(self._highway_layer(fc_outs))
        score_global = self.last_sigm(h_global)
        score_local = tf.concat(scores_list, axis=1)
        score = tf.add(self.alpha * score_global,(1-self.alpha) * score_local)
        return score

class Decoder_HARNN_local(tf.keras.layers.Layer):
    def __init__(self, HIERALCHY, fc_hidden_size, rate, l2_norm=1e-5):
        super(Decoder_HARNN_local, self).__init__()
        NUM_LABELS = 0
        for i in range(len(HIERALCHY)):
            NUM_LABELS += HIERALCHY[i][-1][0]
        self.dense_at1 = [tf.keras.layers.Dense(HIERALCHY[h][-1][0], activation='tanh', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(l2_norm)) for h in range(len(HIERALCHY))]
        self.dense_at2 = [tf.keras.layers.Dense(HIERALCHY[h][-1][0], use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(l2_norm)) for h in range(len(HIERALCHY))]
        self.dense_fc = [tf.keras.layers.Dense(fc_hidden_size, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_norm)) for h in range(len(HIERALCHY))]
        self.dense_lc = [tf.keras.layers.Dense(HIERALCHY[h][-1][0], activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l2_norm)) for h in range(len(HIERALCHY))]
        self.HIERALCHY = HIERALCHY
        #self.dropout = tf.keras.layers.Dropout(rate)

    def _text_attention(self, input_x, h):
        weight = tf.keras.activations.softmax(self.dense_at2[h](self.dense_at1[h](input_x)),axis=-1)
        #out = tf.math.reduce_sum(tf.matmul(weight, input_x),axis=1)
        out = tf.einsum('ijk,ijl->il',weight, input_x)
        return weight, out
    
    def _local_layer(self, input_x, att_weight, h):
        """
        input_x : (batch_size, dim_h)
        att_weight : (batch_size, seq_length, num_cls)
        """
        score = self.dense_lc[h](input_x) #(batch_size, num_cls)
        visual = tf.multiply(att_weight, tf.expand_dims(score, 1)) #(batch_size, seq_length, num_cls)
        visual = tf.nn.softmax(visual)
        visual = tf.reduce_mean(visual, axis=-1) #(batch_size, seq_length)

        return score, visual
            
    def call(self,lstm_out,lstm_pool):
        att_weight, att_out = self._text_attention(lstm_out, 0)
        local_input = tf.concat([lstm_pool, att_out], axis=-1)
        local_fc_out = self.dense_fc[0](local_input)
        fc_outs = [local_fc_out]
        scores, visual = self._local_layer(local_fc_out, att_weight, 0)
        scores_list = [scores]
        for h in range(1, len(self.HIERALCHY)):
            att_input = tf.multiply(lstm_out, tf.expand_dims(visual, -1))
            att_weight, att_out = self._text_attention(att_input, h)
            local_fc_out = self.dense_fc[h](local_input)
            fc_outs.append(local_fc_out)
            scores, visual = self._local_layer(local_fc_out, att_weight, h)
            scores_list.append(scores)
        score_local = tf.concat(scores_list, axis=1)
        return score_local

class Decoder_HARNN_local_list(tf.keras.layers.Layer):
    def __init__(self, HIERALCHY, fc_hidden_size, rate, l2_norm=1e-5):
        super(Decoder_HARNN_local_list, self).__init__()
        NUM_LABELS = 0
        for i in range(len(HIERALCHY)):
            NUM_LABELS += HIERALCHY[i][-1][0]
        self.dense_at1 = [tf.keras.layers.Dense(HIERALCHY[h][-1][0], activation='tanh', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(l2_norm)) for h in range(len(HIERALCHY))]
        self.dense_at2 = [tf.keras.layers.Dense(HIERALCHY[h][-1][0], use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(l2_norm)) for h in range(len(HIERALCHY))]
        self.dense_fc = [tf.keras.layers.Dense(fc_hidden_size, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_norm)) for h in range(len(HIERALCHY))]
        self.dense_lc = [tf.keras.layers.Dense(HIERALCHY[h][-1][0], activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l2_norm)) for h in range(len(HIERALCHY))]
        self.HIERALCHY = HIERALCHY
        #self.dropout = tf.keras.layers.Dropout(rate)

    def _text_attention(self, input_x, h):
        weight = tf.keras.activations.softmax(self.dense_at2[h](self.dense_at1[h](input_x)),axis=-1)
        #out = tf.math.reduce_sum(tf.matmul(weight, input_x),axis=1)
        out = tf.einsum('ijk,ijl->il',weight, input_x)
        return weight, out
    
    def _local_layer(self, input_x, att_weight, h):
        """
        input_x : (batch_size, dim_h)
        att_weight : (batch_size, seq_length, num_cls)
        """
        score = self.dense_lc[h](input_x) #(batch_size, num_cls)
        visual = tf.multiply(att_weight, tf.expand_dims(score, 1)) #(batch_size, seq_length, num_cls)
        visual = tf.nn.softmax(visual)
        visual = tf.reduce_mean(visual, axis=-1) #(batch_size, seq_length)

        return score, visual
            
    def call(self,lstm_out,lstm_pool):
        att_weight, att_out = self._text_attention(lstm_out, 0)
        local_input = tf.concat([lstm_pool, att_out], axis=-1)
        local_fc_out = self.dense_fc[0](local_input)
        fc_outs = [local_fc_out]
        scores, visual = self._local_layer(local_fc_out, att_weight, 0)
        scores_list = [scores]
        for h in range(1, len(self.HIERALCHY)):
            att_input = tf.multiply(lstm_out, tf.expand_dims(visual, -1))
            att_weight, att_out = self._text_attention(att_input, h)
            local_fc_out = self.dense_fc[h](local_input)
            fc_outs.append(local_fc_out)
            scores, visual = self._local_layer(local_fc_out, att_weight, h)
            scores_list.append(scores)

        return scores_list




class HMLN_LSTM(tf.keras.Model):
    def __init__(self,NUM_CLS, NUM_SUBCLS, NUM_MAINGRP, NUM_SUBGRP, NUM_FI,NUM_WORDS,dim_h,r_e,r_h,r_f):
        super(HMLN_LSTM,self).__init__()
        self.embed = tf.keras.layers.Embedding(NUM_WORDS,128,mask_zero=True)
        self.encoder = tf.keras.layers.LSTM(units=dim_h,return_sequences=True,dropout=0.5)
        self.decoder = HMLN(NUM_CLS, NUM_SUBCLS, NUM_MAINGRP, NUM_SUBGRP, NUM_FI,dim_h,r_f)
        self.dropout_e = tf.keras.layers.Dropout(r_e)
        self.dropout_d = tf.keras.layers.Dropout(r_h)

    def call(self,x,training=True):#,y=None):
        em = self.dropout_e(self.embed(x),training=training)
        en_hs = self.dropout_d(self.encoder(em),training=training)
        en_f = en_hs[:,-1,:]
        p_cls,p_subcls,p_maingrp,p_subgrp,p_fi = self.decoder(en_hs,en_f,training=training)
        out = tf.keras.layers.concatenate([p_cls,p_subcls,p_maingrp,p_subgrp,p_fi])
        return out

class HMLN_SET_TRANS(tf.keras.Model):
    def __init__(self,NUM_CLS, NUM_SUBCLS, NUM_MAINGRP, NUM_SUBGRP, NUM_FI,NUM_WORDS,dim_h,r_e,r_h,r_f):
        super(HMLN_SET_TRANS,self).__init__()
        self.encoder = Encoder_SET_TRANS(2, dim_h, 8, dim_h, NUM_WORDS, 500 , rate=r_e)
        self.collector = Collect_outputs(dim_h, 8, dim_h, rate=r_e)
        self.decoder = HMLN(NUM_CLS, NUM_SUBCLS, NUM_MAINGRP, NUM_SUBGRP, NUM_FI,dim_h,r_f)
        self.dropout = tf.keras.layers.Dropout(r_h)
    
    def call(self,x,training=True):
        en_hs = self.encoder(x,training=training,mask = None)
        en_hs = self.dropout(en_hs,training=training)
        en_f = self.collector(en_hs,training=training,mask=None)[:,0]
        p_cls,p_subcls,p_maingrp,p_subgrp,p_fi = self.decoder(en_hs,en_f,training=training)
        out = tf.keras.layers.concatenate([p_cls,p_subcls,p_maingrp,p_subgrp,p_fi])
        return out

class HMLN_RATIO_TRANS(tf.keras.Model):
    def __init__(self,NUM_CLS, NUM_SUBCLS, NUM_MAINGRP, NUM_SUBGRP, NUM_FI,NUM_WORDS,dim_h,num_h,r_e,r_h,r_f):
        super(HMLN_RATIO_TRANS,self).__init__()
        self.encoder = Encoder_RATIO_TRANS(4, dim_h, num_h, dim_h, NUM_WORDS, 500 , rate=r_e)
        self.collector = Collect_outputs(dim_h, num_h, dim_h,rate=r_e)
        self.decoder = HMLN(NUM_CLS, NUM_SUBCLS, NUM_MAINGRP, NUM_SUBGRP, NUM_FI,dim_h,r_f)
        self.dropout = tf.keras.layers.Dropout(r_h)
    
    def call(self,x,training=True):#,y=None):
        en_hs = self.encoder(x,training=training,mask = None)
        en_hs = self.dropout(en_hs,training=training)
        en_f = self.collector(en_hs,training=training,mask=None)[:,0]
        p_cls,p_subcls,p_maingrp,p_subgrp,p_fi = self.decoder(en_hs,en_f,training=training)
        out = tf.keras.layers.concatenate([p_cls,p_subcls,p_maingrp,p_subgrp,p_fi])
        return out

class HSMLN_RATIO_TRANS(tf.keras.Model):
    def __init__(self,NUM_CLS, NUM_SUBCLS, NUM_MAINGRP, NUM_SUBGRP, NUM_FI,ACCUM_H_M1, ACCUM_H_M2, ACCUM_H_M3, ACCUM_H_M4,NUM_WORDS,dim_h,num_h_e,num_h_d,r_e,r_h,r_f):
        super(HSMLN_RATIO_TRANS,self).__init__()
        self.encoder = Encoder_RATIO_TRANS(4, dim_h, num_h_e, dim_h, NUM_WORDS, 500 , rate=r_e)
        self.collector = Collect_outputs(dim_h, num_h_d, dim_h, r_e)
        self.decoder = HSMLN(NUM_CLS, NUM_SUBCLS, NUM_MAINGRP, NUM_SUBGRP, NUM_FI,ACCUM_H_M1, ACCUM_H_M2, ACCUM_H_M3, ACCUM_H_M4,dim_h,num_h_d,r_f)
        self.dropout = tf.keras.layers.Dropout(r_h)
    
    def call(self,x,training=True):#,y=None):
        en_hs = self.encoder(x,training=training,mask = None)
        en_hs = self.dropout(en_hs,training=training)
        en_f = self.collector(en_hs,training=training,mask=None)[:,0]
        p_cls,p_subcls,p_maingrp,p_subgrp,p_fi = self.decoder(en_hs,en_f,training=training)
        out = tf.keras.layers.concatenate([p_cls,p_subcls,p_maingrp,p_subgrp,p_fi])
        return out

class determine(tf.keras.layers.Layer):
    def __init__(self,ACCUM_H_M1, ACCUM_H_M2, ACCUM_H_M3, ACCUM_H_M4):
        super(determine,self).__init__()
        self.ACCUM_H_M1 = ACCUM_H_M1
        self.ACCUM_H_M2 = ACCUM_H_M2
        self.ACCUM_H_M3 = ACCUM_H_M3
        self.ACCUM_H_M4 = ACCUM_H_M4
    
    def call(self,p_cls,p_subcls,p_maingrp,p_subgrp,p_fi):
        p = []
        for i1 in range(len(self.ACCUM_H_M1)-1):
            for i2 in range(self.ACCUM_H_M1[i1],self.ACCUM_H_M1[i1+1]):
                for i3 in range(self.ACCUM_H_M2[i2],self.ACCUM_H_M2[i2+1]):
                    for i4 in range(self.ACCUM_H_M3[i3],self.ACCUM_H_M3[i3+1]):
                        for i5 in range(self.ACCUM_H_M4[i4],self.ACCUM_H_M4[i4+1]):
                            p.append(p_cls[:,i1]*p_subcls[:,i2]*p_maingrp[:,i3]*p_subgrp[:,i4]*p_fi[:,i5])
        p = tf.stack(p,axis=1)
        return p

class HSMLN_RATIO_TRANS_l(tf.keras.Model):
    def __init__(self,NUM_CLS, NUM_SUBCLS, NUM_MAINGRP, NUM_SUBGRP, NUM_FI,ACCUM_H_M1, ACCUM_H_M2, ACCUM_H_M3, ACCUM_H_M4,NUM_WORDS,dim_h,num_h_e,num_h_d,r_e,r_h,r_f):
        super(HSMLN_RATIO_TRANS_l,self).__init__()
        self.encoder = Encoder_RATIO_TRANS(4, dim_h, num_h_e, dim_h, NUM_WORDS, 500 , rate=r_e)
        self.collector = Collect_outputs(dim_h, num_h_d, dim_h, r_e)
        self.decoder = HSMLN(NUM_CLS, NUM_SUBCLS, NUM_MAINGRP, NUM_SUBGRP, NUM_FI,ACCUM_H_M1, ACCUM_H_M2, ACCUM_H_M3, ACCUM_H_M4,dim_h,num_h_d,r_f)
        self.determine = determine(ACCUM_H_M1[:,0], ACCUM_H_M2[:,0], ACCUM_H_M3[:,0], ACCUM_H_M4)
        self.dropout = tf.keras.layers.Dropout(r_h)
    
    def call(self,x,training=True):#,y=None):
        en_hs = self.encoder(x,training=training,mask = None)
        en_hs = self.dropout(en_hs,training=training)
        en_f = self.collector(en_hs,training=training,mask=None)[:,0]
        p_cls,p_subcls,p_maingrp,p_subgrp,p_fi = self.decoder(en_hs,en_f,training=training)
        out = self.determine(p_cls,p_subcls,p_maingrp,p_subgrp,p_fi)
        return out

class DET_PROB(tf.keras.layers.Layer):
    def __init__(self, HIERALCHY):
        super(DET_PROB,self).__init__()
        self.HIERALCHY = HIERALCHY
        self.H = len(HIERALCHY)
    
    def one_step(self, h, i, p_tmp, ps):
        if h == self.H -1:
            for i_ch in range(self.HIERALCHY[h][i][0],self.HIERALCHY[h][i+1][0]):
                p_tmp = p_tmp * self.dc_out[h][:,i_ch]
                ps.append(p_tmp)
            return ps

        else:
            for i_ch in range(self.HIERALCHY[h][i][0],self.HIERALCHY[h][i+1][0]):
                p_tmp = p_tmp * self.dc_out[h][:,i_ch]
                ps = self.one_step(h+1, i_ch, p_tmp, ps)
            return ps
    
    def call(self,decoder_output):
        self.dc_out = decoder_output
        p = self.one_step(0, 0, 1., [])
        p = tf.stack(p,axis=1)
        return p


class HSMLN_RATIO_TRANS_H(tf.keras.Model):
    def __init__(self,HIERALCHY,NUM_WORDS,dim_h,num_h_e,num_h_d,r_e,r_h,r_f, embedding_matrix):
        super(HSMLN_RATIO_TRANS_H,self).__init__()
        NUM_WORDS,d_model = embedding_matrix.shape
        self.encoder = Encoder_RATIO_TRANS(4, d_model, num_h_e, dim_h, NUM_WORDS, embedding_matrix, 500 , rate=r_e)
        self.collector = Collect_outputs(dim_h, num_h_d, dim_h, r_e)
        self.decoder = HSMLN(HIERALCHY,dim_h,num_h_d,rate=r_f)
        self.det = DET_PROB(HIERALCHY)
        self.dropout = tf.keras.layers.Dropout(r_h)
    
    def call(self,x,training=True):#,y=None):
        en_hs = self.encoder(x)
        en_hs = self.dropout(en_hs,training=training)
        en_f = self.collector(en_hs,mask=None)[:,0]
        ps = self.decoder(en_hs,en_f)
        out = self.det(ps)
        return out

class BASELINE(tf.keras.Model):
    def __init__(self,NUM_FI,NUM_WORDS,dim_h,num_h_e):
        super(BASELINE,self).__init__()
        self.encoder = Encoder_RATIO_TRANS(4, dim_h, num_h_e, dim_h, NUM_WORDS, 500 ,0.)
        self.collector = Collect_outputs(dim_h, num_h_e, dim_h,0.)
        self.decoder = Decoder_Base(NUM_FI)
    
    def call(self,x,training=True):#,y=None):
        en_hs = self.encoder(x,training=training,mask = None)
        en_f = self.collector(en_hs,training=training,mask=None)[:,0]
        p = self.decoder(en_f,training=training)
        return p


class HARNN(tf.keras.Model):
    def __init__(self,HIERALCHY,NUM_WORDS,dim_h,d_model,num_units,rate,embedding_matrix=None):
        super(HARNN,self).__init__()
        if embedding_matrix is None:
            self.embedd = tf.keras.layers.Embedding(NUM_WORDS, d_model,mask_zero=True)
        else:
            self.embedd = tf.keras.layers.Embedding(NUM_WORDS, d_model,mask_zero=True, weights = [embedding_matrix])
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(num_units, return_sequences=True))
        self.decoder = Decoder_HARNN(HIERALCHY, dim_h, rate, alpha=0.5, l2_norm=1e-5)
        self.dropout = tf.keras.layers.Dropout(rate)
  
    def call(self,x):
        embed = self.embedd(x)
        out_lstm = self.bilstm(self.dropout(embed))
        pool_lstm = tf.math.reduce_mean(out_lstm, axis=1)
        out = self.decoder(out_lstm, pool_lstm)
        return out

class TRANSFORMER_UNIT(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, dff, rate, epsilon_norm = 1e-6):
        super(TRANSFORMER_UNIT, self).__init__()
        self.MHA = tf.keras.layers.MultiHeadAttention(num_heads, key_dim)
        self.ffn1 = tf.keras.layers.Dense(dff, activation='relu')
        self.ffn2 = tf.keras.layers.Dense(key_dim)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=epsilon_norm)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=epsilon_norm)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    
    def call(self,x):
        h1 = self.MHA(x, x)
        h2 = self.layernorm1(self.dropout1(h1) + x)
        h3 = self.ffn2(self.ffn1(h2))
        out = self.layernorm2(self.dropout2(h3) + x)
        return out

class ENCODER_TRANS(tf.keras.Model):
    def __init__(self, d_model, num_it, num_heads, rate):
        super(ENCODER_TRANS, self).__init__()
        self.trans_unit = [TRANSFORMER_UNIT(num_heads, d_model, d_model, rate) for i in range(num_it)]
        self.dropout = tf.keras.layers.Dropout(rate)
        
    def call(self,x):
        for i in range(len(self.trans_unit)):
            x = self.trans_unit[i](x)
        return x

class RATIO_EMBEDDING(tf.keras.Model):
    def __init__(self, input_vocab_size, d_model, embedding_matrix):
        super(RATIO_EMBEDDING, self).__init__()
        self.d_model = d_model

        if embedding_matrix is None:
          self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model,mask_zero=True)
        else:
          self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model,mask_zero=True, weights = [embedding_matrix])
        
        self.mask0 = tf.keras.layers.Masking(mask_value=0)
        self.expand = Lambda(lambda x: K.expand_dims(x,axis=2))
    

    def call(self, x):
        words = x[:,0]
        ratio = x[:,1]
        ratio = self.expand(self.mask0(ratio))
        em = self.embedding(words)
        out = ratio * em
        out *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        
        return out


class HARNN_RTRANS(tf.keras.Model):
    def __init__(self,HIERALCHY,NUM_WORDS,dim_h,d_model,rate,embedding_matrix=None, num_h_e= 8, num_it=2):
        super(HARNN_RTRANS,self).__init__()
        self.embedd = RATIO_EMBEDDING(NUM_WORDS, d_model, embedding_matrix)
        self.encoder = ENCODER_TRANS(d_model, num_it, num_h_e, rate)
        #self.collector = Collect_outputs(dim_h, num_h_e, dim_h, rate)
        self.decoder = Decoder_HARNN(HIERALCHY, dim_h, rate, alpha=0.5, l2_norm=1e-5)
        self.dropout = tf.keras.layers.Dropout(rate)
  
    def call(self,x):
        em = self.embedd(x)
        en_hs = self.encoder(em)
        en_f = tf.math.reduce_mean(en_hs, axis=1)
        en_hs = self.dropout(en_hs)
        #en_f = self.collector(en_hs,mask=None)[:,0]
        out = self.decoder(en_hs, en_f)
        return out

class HARNN_local_RTRANS(tf.keras.Model):
    def __init__(self,HIERALCHY,NUM_WORDS,dim_h,d_model,rate,embedding_matrix=None, num_h_e= 8, num_it=2):
        super(HARNN_local_RTRANS,self).__init__()
        self.embedd = RATIO_EMBEDDING(NUM_WORDS, d_model, embedding_matrix)
        self.encoder = ENCODER_TRANS(d_model, num_it, num_h_e, rate)
        #self.collector = Collect_outputs(dim_h, num_h_e, dim_h, rate)
        self.decoder = Decoder_HARNN_local(HIERALCHY, dim_h, rate, l2_norm=1e-5)
        self.dropout = tf.keras.layers.Dropout(rate)
  
    def call(self,x):
        em = self.embedd(x)
        en_hs = self.encoder(em)
        en_f = tf.math.reduce_mean(en_hs, axis=1)
        en_hs = self.dropout(en_hs)
        #en_f = self.collector(en_hs,mask=None)[:,0]
        out = self.decoder(en_hs, en_f)
        return out

class HARNN_local_RTRANS_conditional(tf.keras.Model):
    def __init__(self,HIERALCHY,NUM_WORDS,dim_h,d_model,rate,embedding_matrix=None, num_h_e= 8, num_it=2):
        super(HARNN_local_RTRANS_conditional,self).__init__()
        self.embedd = RATIO_EMBEDDING(NUM_WORDS, d_model, embedding_matrix)
        self.encoder = ENCODER_TRANS(d_model, num_it, num_h_e, rate)
        #self.collector = Collect_outputs(dim_h, num_h_e, dim_h, rate)
        self.decoder = Decoder_HARNN_local_list(HIERALCHY, dim_h, rate, l2_norm=1e-5)
        self.detprob = DET_PROB(HIERALCHY)
        self.dropout = tf.keras.layers.Dropout(rate)
  
    def call(self,x):
        em = self.embedd(x)
        en_hs = self.encoder(em)
        en_f = tf.math.reduce_mean(en_hs, axis=1)
        en_hs = self.dropout(en_hs)
        #en_f = self.collector(en_hs,mask=None)[:,0]
        out = self.detprob(self.decoder(en_hs, en_f))
        return out

class HARNN_TRANS(tf.keras.Model):
    def __init__(self,HIERALCHY,NUM_WORDS,dim_h,d_model,rate,embedding_matrix=None, num_h_e= 8, num_it=2):
        super(HARNN_TRANS,self).__init__()
        self.embedd = tf.keras.layers.Embedding(NUM_WORDS, d_model,mask_zero=True, weights = [embedding_matrix])
        self.encoder = ENCODER_TRANS(d_model, num_it, num_h_e, rate)
        #self.collector = Collect_outputs(dim_h, num_h_e, dim_h, rate)
        self.decoder = Decoder_HARNN(HIERALCHY, dim_h, rate, alpha=0.5, l2_norm=1e-5)
        self.dropout = tf.keras.layers.Dropout(rate)
  
    def call(self,x):
        em = self.embedd(x)
        en_hs = self.encoder(em)
        en_f = tf.math.reduce_mean(en_hs, axis=1)
        en_hs = self.dropout(en_hs)
        #en_f = self.collector(en_hs,mask=None)[:,0]
        out = self.decoder(en_hs, en_f)
        return out

class HSMLN_RTRANS(tf.keras.Model):
    def __init__(self,HIERALCHY,NUM_WORDS,dim_h,d_model,rate,embedding_matrix=None, num_h_e= 8, num_it=2):
        super(HSMLN_RTRANS,self).__init__()
        self.embedd = RATIO_EMBEDDING(NUM_WORDS, d_model, embedding_matrix)
        #self.encoder = ENCODER_TRANS(d_model, num_it, num_h_e, rate)
        self.encoder = ENCODER_TRANS(dim_h, num_it, num_h_e, rate)
        print("s")
        #self.collector = Collect_outputs(dim_h, num_h_e, dim_h, rate)
        self.decoder = HSMLN(HIERALCHY,dim_h,num_h_e,rate=rate)
        self.det = DET_PROB(HIERALCHY)
        #self.decoder = Decoder_HARNN(HIERALCHY, dim_h, rate, alpha=0.5, l2_norm=1e-5)
        self.dropout = tf.keras.layers.Dropout(rate)
  
    def call(self,x):
        em = self.embedd(x)
        en_hs = self.encoder(em)
        en_f = tf.math.reduce_mean(en_hs, axis=1)
        en_hs = self.dropout(en_hs)
        ps = self.decoder(en_hs,en_f)
        out = self.det(ps)
        return out

class HSMLN_RTRANS_outputall(tf.keras.Model):
    def __init__(self,HIERALCHY,NUM_WORDS,dim_h,d_model,rate,embedding_matrix=None, num_h_e= 8, num_it=2):
        super(HSMLN_RTRANS_outputall,self).__init__()
        self.embedd = RATIO_EMBEDDING(NUM_WORDS, d_model, embedding_matrix)
        self.encoder = ENCODER_TRANS(d_model, num_it, num_h_e, rate)
        self.decoder = HSMLN(HIERALCHY,dim_h,num_h_e,rate=rate)
        self.dropout = tf.keras.layers.Dropout(rate)
  
    def call(self,x):
        em = self.embedd(x)
        en_hs = self.encoder(em)
        en_f = tf.math.reduce_mean(en_hs, axis=1)
        en_hs = self.dropout(en_hs)
        ps = self.decoder(en_hs,en_f)
        ps = tf.concat(ps,axis=1)
        return ps

class Multilabel_RTRANS(tf.keras.Model):
    def __init__(self,NUM_LABEL,NUM_WORDS,d_model,rate,embedding_matrix=None, num_h_e= 8, num_it=2):
        super(Multilabel_RTRANS,self).__init__()
        self.embedd = RATIO_EMBEDDING(NUM_WORDS, d_model, embedding_matrix)
        self.encoder = ENCODER_TRANS(d_model, num_it, num_h_e, rate)
        self.decoder = tf.keras.layers.Dense(NUM_LABEL, activation="sigmoid")
  
    def call(self,x):
        em = self.embedd(x)
        en_hs = self.encoder(em)
        en_f = tf.math.reduce_mean(en_hs, axis=1)
        ps = self.decoder(en_f)
        return ps

class Multilabel_TRANS(tf.keras.Model):
    def __init__(self,NUM_LABEL,NUM_WORDS,d_model,rate,embedding_matrix=None, num_h_e= 8, num_it=2, max_length=500):
        super(Multilabel_TRANS,self).__init__()
        self.embedd = tf.keras.layers.Embedding(NUM_WORDS, d_model,mask_zero=True, weights = [embedding_matrix])
        self.pos_encoding = positional_encoding(max_length,d_model)
        self.encoder = ENCODER_TRANS(d_model, num_it, num_h_e, rate)
        self.decoder = tf.keras.layers.Dense(NUM_LABEL, activation="sigmoid")
        self.d_model = d_model
  
    def call(self,x):
        seq_len = tf.shape(x)[1]
        em = self.embedd(x)
        em *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        em += self.pos_encoding[:, :seq_len, :]
        en_hs = self.encoder(em)
        en_f = tf.math.reduce_mean(en_hs, axis=1)
        ps = self.decoder(en_f)
        return ps

class CLOSS(tf.keras.losses.Loss):
    def __init__(self, ACCUM_H_M1, ACCUM_H_M2, ACCUM_H_M3, ACCUM_H_M4,NUM_HIER):
        super(CLOSS,self).__init__()
        self.ACCUM_H_M1 = ACCUM_H_M1
        self.ACCUM_H_M2 = ACCUM_H_M2
        self.ACCUM_H_M3 = ACCUM_H_M3
        self.ACCUM_H_M4 = ACCUM_H_M4
        self.NUM_HIER = NUM_HIER
    
    def prob_cond(self, L_cls, L_subcls, L_maingrp, L_subgrp,L_fi, t1,t2,t3,t4,t5):
        p,t = [],[]
        for i1 in range(len(self.ACCUM_H_M1)-1):
            for i2 in range(self.ACCUM_H_M1[i1],self.ACCUM_H_M1[i1+1]):
                for i3 in range(self.ACCUM_H_M2[i2],self.ACCUM_H_M2[i2+1]):
                    for i4 in range(self.ACCUM_H_M3[i3],self.ACCUM_H_M3[i3+1]):
                        for i5 in range(self.ACCUM_H_M4[i4],self.ACCUM_H_M4[i4+1]):
                            p.append(L_cls[:,i1]*L_subcls[:,i2]*L_maingrp[:,i3]*L_subgrp[:,i4]*L_fi[:,i5])
                            t.append(t5[:,i5])
        p = tf.stack(p,axis=1)
        t = tf.stack(t,axis=1)
        return tf.keras.losses.binary_crossentropy(t,p)
    
    def call(self,y,p):
        NUM_HIER = self.NUM_HIER
        return self.prob_cond(p[:,0:NUM_HIER[0]],p[:,NUM_HIER[0]:NUM_HIER[1]],p[:,NUM_HIER[1]:NUM_HIER[2]],p[:,NUM_HIER[2]:NUM_HIER[3]],p[:,NUM_HIER[3]:NUM_HIER[4]],
                              y[:,0:NUM_HIER[0]],y[:,NUM_HIER[0]:NUM_HIER[1]],y[:,NUM_HIER[1]:NUM_HIER[2]],y[:,NUM_HIER[2]:NUM_HIER[3]],y[:,NUM_HIER[3]:NUM_HIER[4]])

class CLOSS_l(tf.keras.losses.Loss):    
    def call(self,y,p):
        return tf.keras.losses.binary_crossentropy(y,p)

class FOCAL_LOSS(tf.keras.losses.Loss):
    def __init__(self,gamma = 2.,epsilon=1e-6):
        super(FOCAL_LOSS,self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon
    
    def call(self,y_true,y_pred):
        softmax_loss = -tf.reduce_sum(y_true*((1-y_pred)**self.gamma) * tf.math.log(y_pred+self.epsilon),axis=-1)#focal_loss
        return softmax_loss



class HLOSS(tf.keras.losses.Loss):
    def __init__(self, ACCUM_H_M1, ACCUM_H_M2, ACCUM_H_M3, ACCUM_H_M4, NUM_HIER, LAMBDA):
        super(HLOSS,self).__init__()
        self.relu = tf.keras.layers.ReLU()
        self.ACCUM_H_M1 = ACCUM_H_M1
        self.ACCUM_H_M2 = ACCUM_H_M2
        self.ACCUM_H_M3 = ACCUM_H_M3
        self.ACCUM_H_M4 = ACCUM_H_M4
        self.NUM_HIER = NUM_HIER
        self.LAMBDA = LAMBDA
    
    def GLL(self, L_cls, L_subcls, L_maingrp, L_subgrp,L_fi, t1,t2,t3,t4,t5):
        LL_fi = tf.keras.losses.binary_crossentropy(t5,L_fi)
        return LL_fi

    def LH_LOSS(self, parent, child, p_c):
        new_child = []
        for i in range(len(parent[0])):
            temp = tf.reduce_max(child[:,p_c[i]:p_c[i+1]],axis=-1)
            new_child.append(temp)
        new_child =tf.stack(new_child,axis=1)
        return new_child # LAMBDA is hyperparameter that indicates weight of H_LOSS

    def H_LOSS(self, L_cls, L_subcls, L_maingrp, L_subgrp,L_fi):
        HM_cls = self.LH_LOSS(L_cls,L_subcls,self.ACCUM_H_M1)
        HM_subcls = self.LH_LOSS(L_subcls,L_maingrp,self.ACCUM_H_M2)
        HM_maingrp = self.LH_LOSS(L_maingrp,L_subgrp,self.ACCUM_H_M3)
        HM_subgrp = self.LH_LOSS(L_subgrp,L_fi,self.ACCUM_H_M4)
        HL_cls = tf.math.reduce_sum(self.relu(-1*L_cls + HM_cls)**2,axis = -1)
        HL_subcls = tf.math.reduce_sum(self.relu(-1*L_subcls + HM_subcls)**2,axis=-1)
        HL_maingrp = tf.math.reduce_sum(self.relu(-1*L_maingrp + HM_maingrp)**2,axis=-1)
        HL_subgrp = tf.math.reduce_sum(self.relu(-1*L_subgrp + HM_subgrp)**2,axis=-1)
        HL_ALL = (HL_cls + HL_subcls + HL_maingrp+HL_subgrp)*self.LAMBDA
        return HL_ALL
    
    def call(self,y,p):
        NUM_HIER = self.NUM_HIER
        m_loss = self.GLL(p[:,0:NUM_HIER[0]],p[:,NUM_HIER[0]:NUM_HIER[1]],p[:,NUM_HIER[1]:NUM_HIER[2]],p[:,NUM_HIER[2]:NUM_HIER[3]],p[:,NUM_HIER[3]:NUM_HIER[4]],
                          y[:,0:NUM_HIER[0]],y[:,NUM_HIER[0]:NUM_HIER[1]],y[:,NUM_HIER[1]:NUM_HIER[2]],y[:,NUM_HIER[2]:NUM_HIER[3]],y[:,NUM_HIER[3]:NUM_HIER[4]])
        h_loss = self.H_LOSS(p[:,0:NUM_HIER[0]],p[:,NUM_HIER[0]:NUM_HIER[1]],p[:,NUM_HIER[1]:NUM_HIER[2]],p[:,NUM_HIER[2]:NUM_HIER[3]],p[:,NUM_HIER[3]:NUM_HIER[4]])
        return tf.math.add_n([m_loss,h_loss])
