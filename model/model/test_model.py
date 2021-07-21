from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, concatenate, LeakyReLU, Cropping2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def UNet():
    main_input = Input(shape=(256, 256, 5), name='img_input')

    ''' ~~~~~~~~~~~~~~~~~~~ ENCODING LAYERS ~~~~~~~~~~~~~~~~~~~ '''

    c1 = Conv2D(32, kernel_size=(3,3), padding = 'same')(main_input)
    c1 = LeakyReLU(0.2)(c1)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(32, kernel_size=(3,3), kernel_regularizer=l2(0.01), padding = 'same')(c1)
    c1 = LeakyReLU(0.2)(c1)
    c1 = BatchNormalization()(c1)

    print("c1: "+str(c1.shape))

    p1 = MaxPooling2D((2,2))(c1)

    print("p1: "+str(p1.shape))

    c2 = Conv2D(32*2, kernel_size=(3,3), padding = 'same')(p1)
    c2 = LeakyReLU(0.2)(c2)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(32*2, kernel_size=(3,3), padding = 'same')(c2)
    c2 = LeakyReLU(0.2)(c2)
    c2 = BatchNormalization()(c2)

    print("c2: "+str(c2.shape))

    p2 = MaxPooling2D((2,2))(c2)

    print("p2: "+str(p2.shape))

    c3 = Conv2D(32*4, kernel_size=(3,3), padding = 'same')(p2)
    c3 = LeakyReLU(0.2)(c3)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(32*2, kernel_size=(1,1), padding = 'same')(c3)
    c3 = LeakyReLU(0.2)(c3)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(32*4, kernel_size=(3,3), padding = 'same')(c3)
    c3 = LeakyReLU(0.2)(c3)
    c3 = BatchNormalization()(c3)

    print("c3: "+str(c3.shape))

    p3 = MaxPooling2D((2,2))(c3)

    print("p3: "+str(p3.shape))

    c4 = Conv2D(32*6, kernel_size=(3,3), padding = 'same')(p3)
    c4 = LeakyReLU(0.2)(c4)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(32*4, kernel_size=(1,1), padding = 'same')(c4)
    c4 = LeakyReLU(0.2)(c4)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(32*6, kernel_size=(3,3), padding = 'same')(c4)
    c4 = LeakyReLU(0.2)(c4)
    c4 = BatchNormalization()(c4)

    print("c4: "+str(c4.shape))

    p4 = MaxPooling2D((2,2))(c4)

    print("p4: "+str(p4.shape))

    c5 = Conv2D(32*6, kernel_size=(3,3), padding = 'same')(p4)
    c5 = LeakyReLU(0.2)(c5)
    c5 = BatchNormalization()(c5)

    print("c5: "+str(c5.shape))


    ''' ~~~~~~~~~~~~~~~~~~~ DECODING LAYERS ~~~~~~~~~~~~~~~~~~~ '''

    u1 = UpSampling2D((2,2))(c5)
    print("u1: "+str(u1.shape))
    concat1 = concatenate([u1, Cropping2D(((0,1), (0,0)))(c4)])
    print("concat1: "+str(concat1.shape))
    c6 = Conv2D(32*4, kernel_size=(3,3), padding = 'same')(concat1)
    c6 = LeakyReLU(0.2)(c6)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(32*4, kernel_size=(3,3), padding = 'same')(c6)
    c6 = LeakyReLU(0.2)(c6)
    c6 = BatchNormalization()(c6)

    print("c6: "+str(c6.shape))

    u2 = UpSampling2D((2,2))(c6)
    print("u2: "+str(u2.shape))
    concat2 = concatenate([Cropping2D(((1,1), (0,0)))(c3), u2])
    print("concat2: "+str(concat2.shape))
    c7 = Conv2D(32*2, kernel_size=(3,3), padding = 'same')(concat2)
    c7 = LeakyReLU(0.2)(c7)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(32*2, kernel_size=(3,3), padding = 'same')(c7)
    c7 = LeakyReLU(0.2)(c7)
    c7 = BatchNormalization()(c7)

    print("c7: "+str(c7.shape))

    u3 = UpSampling2D((2,2))(c7)
    print("u3: "+str(u3.shape))
    concat3 = concatenate([Cropping2D(((2,2), (0,0)))(c2), u3])
    print("concat3: "+str(concat3.shape))

    c8 = Conv2D(32, kernel_size=(3,3), padding = 'same')(concat3)
    c8 = LeakyReLU(0.2)(c8)
    c8 = BatchNormalization()(c8)
    c8 = Conv2D(32, kernel_size=(3,3), padding = 'same')(c8)
    c8 = LeakyReLU(0.2)(c8)
    c8 = BatchNormalization()(c8)

    print("c8: "+str(c8.shape))

    u4 = UpSampling2D((2,2))(c8)
    print("u4: "+str(u4.shape))
    concat4 = concatenate([Cropping2D(((4,4),(0,0)))(c1), u4])
    print("concat4: "+str(concat4.shape))

    c9 = Conv2D(16, kernel_size=(1,1), padding='same')(concat4)
    c9 = LeakyReLU(0.2)(c9)
    c9 = BatchNormalization()(c9)
    print("c9: "+str(c9.shape))

    mask_out = Conv2D(4, (1,1), padding='same', activation='sigmoid', name='mask_out')(c9)
    print("mask_out: "+str(mask_out.shape))

    return Model(inputs=[main_input], outputs=[mask_out])


# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
# model.summary()