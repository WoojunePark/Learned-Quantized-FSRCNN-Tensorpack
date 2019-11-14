import tensorflow as tf
sess = tf.InteractiveSession()


nbit = 3
num_levels = 2**nbit

init_level_multiplier = []
for i in range(0, num_levels):
    level_multiplier_i = [9. for j in range(nbit)]
    # print(level_multiplier_i)
    # print("----------------------")

    level_number = i
    for j in range(nbit):
        level_multiplier_i[j] = float(level_number % 2)
        level_number = level_number // 2
    init_level_multiplier.append(level_multiplier_i)
    # print(init_level_multiplier)


test = [1, 2, 3, 4, 5]
test = tf.constant(test)
test_t = tf.transpose(test)

levels, sort_id = tf.math.top_k(test_t, 5)
levels = tf.print(levels, [levels], message="\"")

sort_id = tf.print(sort_id, [sort_id], message="\"")
