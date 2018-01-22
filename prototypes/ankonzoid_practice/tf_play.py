import tensorflow as tf

sess = tf.Session()

if 0:
    a = tf.constant(5,name="a")
    b = tf.constant(15,name="b")
    c = tf.add(a,b,name="c")
    output = sess.run(c)
    print(output)

if 1:
    n_input = 2
    n_output = 2
    n_hidden_1 = 20
    x_input = tf.placeholder(tf.float32, (None, n_input), name="states")
    W1 = tf.get_variable("W1", [n_input, n_hidden_1], initializer=tf.random_normal_initializer())
    b1 = tf.get_variable("b1", [n_hidden_1], initializer=tf.constant_initializer(0))
    h1 = tf.nn.tanh(tf.matmul(x_input, W1) + b1)  # first layer: tanh(s*W1+b1)
    W2 = tf.get_variable("W2", [n_hidden_1, n_output], initializer=tf.random_normal_initializer(stddev=0.1))
    b2 = tf.get_variable("b2", [n_output], initializer=tf.constant_initializer(0))
    x_output = tf.matmul(h1, W2) + b2  # second layer: s*W2+b2

    var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)  # collect variable list
    sess.run(tf.variables_initializer(var_lists))  # initializer variables
    sess.run(tf.assert_variables_initialized())  # make sure variables have been initialized

    x_example = [[100,100]]
    output = sess.run(x_output, {x_input: x_example})[0]
    print(output)


sess.close()