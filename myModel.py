import os, gc,time 
import tensorflow as tf
from sklearn.model_selection import train_test_split

class Mymodel(tf.keras.Model):

    def __init__(self,  vocab_size=5000 , 
                        max_sequence_length=25,
                        embedding_dim = 256,
                        number_lstm_units = 256,
                        number_dense_units = 256,
                        rate_drop_lstm = 0.17 , 
                        rate_drop_dense = 0.01):
        super(Mymodel, self).__init__()            
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.number_lstm_units = number_lstm_units
        self.number_dense_units = number_dense_units
        self.tokenzierCreated = False
        self.rate_drop_lstm = rate_drop_lstm
        self.rate_drop_dense = rate_drop_dense
        self.freature_extratcor= tf.keras.applications.InceptionV3(include_top=False,weights='imagenet')
        self.validation_split_ratio =0.2
        self.model_save_directory='.'
        
    def prepare_tokenizer(self,documents):
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.vocab_size,
                                                                oov_token="<unk>",
                                                                filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
        self.tokenizer.fit_on_texts(documents)
        self.tokenizer.word_index['<pad>'] = 0
        self.tokenizer.index_word[0] = '<pad>'
        self.tokenzierCreated = True

    
        
    def model(self):
        sequence_1_input = tf.keras.layers.Input(shape=(self.max_sequence_length,), dtype='int32')
        feature_2_input = tf.keras.layers.Input(shape=(8*8*2048,))

        embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)
        lstm_layer = tf.keras.layers.Bidirectional( tf.keras.layers.LSTM(
                            self.number_lstm_units, dropout=self.rate_drop_lstm, 
                            recurrent_dropout=self.rate_drop_lstm))
        fully_connected_in=tf.keras.layers.Dense(self.embedding_dim)
        fully_connected_out = tf.nn.relu(fully_connected_in(feature_2_input))

        # image_features=self.freature_extratcor(feature_2_input)
        # feature_embeddings=fully_connected(feature_2_input)
        lstm_input1=embedding(sequence_1_input)
        lstm_output1=lstm_layer(lstm_input1)
        lstm_input2=embedding(fully_connected_out)
        lstm_output2=lstm_layer(lstm_input2)
        merged = tf.keras.layers.concatenate([lstm_output1, lstm_output2])
        merged = tf.keras.layers.BatchNormalization()(merged)
        merged = tf.keras.layers.Dropout(self.rate_drop_dense)(merged)
        merged = tf.keras.layers.Dense(self.number_dense_units, activation= 'relu')(merged)
        merged = tf.keras.layers.BatchNormalization()(merged)
        merged = tf.keras.layers.Dropout(self.rate_drop_dense)(merged)
        preds = tf.keras.layers.Dense(1, activation='sigmoid')(merged)
        model = tf.keras.Model(inputs=[sequence_1_input, feature_2_input], outputs=preds)
        return model

    def dataPost_processing(self,imagePath, text):
        for index,image_path in enumerate(imagePath):
            img = tf.io.read_file(image_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, (299, 299))
            img = tf.keras.applications.inception_v3.preprocess_input(tf.expand_dims(img,0))
            img=self.freature_extratcor(img)
            img = tf.reshape(img, (img.shape[0],img.shape[1]*img.shape[2]*img.shape[3]))
            if index == 0 :
                img_features = img
            else:
                img_features = tf.concat([img_features,img],0)        
        
        text_seqs = self.tokenizer.texts_to_sequences(text)
        text_vector=tf.keras.preprocessing.sequence.pad_sequences(text_seqs, padding='post', 
                maxlen=self.max_sequence_length )
        return img_features, text_vector        

    def train(self,imagePath,dictonary,is_similar):
        imagePath_train, imagePath_val, dictonary_train, dictonary_val,is_similar_train, is_similar_val = \
        train_test_split(imagePath,dictonary,is_similar,test_size=0.2,random_state=0)
        img_features_train, text_vector_train =self.dataPost_processing(imagePath_train,dictonary_train)
        gc.collect()
        img_features_val, text_vector_val =self.dataPost_processing(imagePath_val,dictonary_val)
        gc.collect()
        model = self.model()  
        model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])
        STAMP = 'lstm_%d_%d_%.2f_%.2f' % (self.number_lstm_units, self.number_dense_units, self.rate_drop_lstm, self.rate_drop_dense)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        checkpoint_dir = self.model_save_directory + 'checkpoints/' + str(int(time.time())) + '/'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)  
        bst_model_path = checkpoint_dir + STAMP + '.h5'
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)

        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=checkpoint_dir + "logs/{}".format(time.time()))

        model.fit([img_features_val, text_vector_train], is_similar_train,
                  validation_data=([val_data_x1, val_data_x2, leaks_val], is_similar_val),
                  epochs=200, batch_size=16, shuffle=True,
                  callbacks=[early_stopping, model_checkpoint, tensorboard])

        return bst_model_path