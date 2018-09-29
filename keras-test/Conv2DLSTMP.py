import tensorflow as tf
from keras.layers import ConvLSTM2DCell, K


class Conv2DLSTMP(ConvLSTM2DCell):
    def call(self, inputs, states, training=None):
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = self._generate_dropout_mask(
                K.ones_like(inputs),
                self.dropout,
                training=training,
                count=4)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = self._generate_dropout_mask(
                K.ones_like(states[1]),
                self.recurrent_dropout,
                training=training,
                count=4)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state
        shape=dp_mask[0].shape
        if(inputs.shape==shape):
           pass
        else:
           dp_mask=tf.slice(dp_mask,[0,0,0,0,0],[4,shape[0]//2,shape[1],shape[2],shape[3]])

        shape=rec_dp_mask[0].shape
        if(inputs.shape[0]==shape[0]):
           pass
        else:
           rec_dp_mask=tf.slice(rec_dp_mask,[0,0,0,0,0],[4,shape[0]//2,shape[1],shape[2],shape[3]])

        if 0 < self.dropout < 1.:
            inputs_i = inputs * dp_mask[0]
            inputs_f = inputs * dp_mask[1]
            inputs_c = inputs * dp_mask[2]
            inputs_o = inputs * dp_mask[3]
        else:
            inputs_i = inputs
            inputs_f = inputs
            inputs_c = inputs
            inputs_o = inputs

        if 0 < self.recurrent_dropout < 1.:
            h_tm1_i = h_tm1 * rec_dp_mask[0]
            h_tm1_f = h_tm1 * rec_dp_mask[1]
            h_tm1_c = h_tm1 * rec_dp_mask[2]
            h_tm1_o = h_tm1 * rec_dp_mask[3]
        else:
            h_tm1_i = h_tm1
            h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            h_tm1_o = h_tm1

        x_i = self.input_conv(inputs_i, self.kernel_i, self.bias_i,
                              padding=self.padding)
        x_f = self.input_conv(inputs_f, self.kernel_f, self.bias_f,
                              padding=self.padding)
        x_c = self.input_conv(inputs_c, self.kernel_c, self.bias_c,
                              padding=self.padding)
        x_o = self.input_conv(inputs_o, self.kernel_o, self.bias_o,
                              padding=self.padding)
        h_i = self.recurrent_conv(h_tm1_i,
                                  self.recurrent_kernel_i)
        h_f = self.recurrent_conv(h_tm1_f,
                                  self.recurrent_kernel_f)
        h_c = self.recurrent_conv(h_tm1_c,
                                  self.recurrent_kernel_c)
        h_o = self.recurrent_conv(h_tm1_o,
                                  self.recurrent_kernel_o)

        i = self.recurrent_activation(x_i + h_i)
        f = self.recurrent_activation(x_f + h_f)
        c = f * c_tm1 + i * self.activation(x_c + h_c)
        o = self.recurrent_activation(x_o + h_o)
        h = o * self.activation(c)

        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True

        return h, [h, c]