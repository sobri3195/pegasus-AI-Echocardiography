class classifier(object):
    def __init__(self, img_rows = 120, img_cols = 160, batch_size = 512, epochs = 100):
        self.img_rows = img_rows
        self.img_cols = img_cols

		self.batch_size = batch_size
		self.epochs = epochs
		self.lr_init = 0.02
		self.lr_decay = 0.85
		self.optimizer = Adam(lr=self.lr_init, decay=self.lr_decay)
		self.metrics = ['accuracy']
		self.loss = 'categorical_crossentropy'

	def classifier_model(self):
		model = Sequential()

		model.add(Conv2D(16, (3, 3), padding='same', input_shape=(self.img_rows, self.img_cols, 1)))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model = Conv2DBatchNormRelu(model, 16, (3, 3), padding='same')
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model = Conv2DBatchNormRelu(model, 32, (3, 3), padding='same')
		model = Conv2DBatchNormRelu(model, 32, (3, 3), padding='same')
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model = Conv2DBatchNormRelu(model, 64, (3, 3), padding='same')
		model = Conv2DBatchNormRelu(model, 64, (3, 3), padding='same')
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model = Conv2DBatchNormRelu(model, 128, (3, 3), padding='same')
		model = Conv2DBatchNormRelu(model, 128, (3, 3), padding='same')
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Flatten())
		model.add(Dense(1024, kernel_regularizer=l2(0.02)))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(Dropout(0.5))
		model.add(Dense(512, kernel_regularizer=l2(0.02)))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(Dropout(0.5))
		model.add(Dense(15, activation='softmax'))

		model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
	
		return model

def Conv2DBatchNormRelu(model, filters, kernel_size, padding='valid', strides=(1, 1), name=None):
    model.add(Conv2D(filters, kernel_size, padding=padding ,strides=strides, use_bias=False, name=name))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    return model

def DenseBatchNormRelu(model, units, name=None):
    model.add(Dense(units, use_bias=False, name=name))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    return model


class segmenter(object):
    def __init__(self, img_rows = 120, img_cols = 160):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def get_unet(self):
        inputs = Input((self.img_rows, self.img_cols,1))
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)

        
        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop4))
        

        merge6 = merge([conv3,up6], mode = 'concat', concat_axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

        model = Model(input = inputs, output = conv10)

        model.compile(optimizer = Adam(lr = 2e-5), loss = 'binary_crossentropy', metrics = ['accuracy'])

        return model
