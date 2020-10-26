import sys

import tensorflow as tf

import model

def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    return tf.cond(
       tf.equal(k, 0),
       lambda: logits,
       lambda: _top_k(),
    )


def top_p_logits(logits, p):
    with tf.variable_scope('top_p_logits'):
        logits_sort = tf.sort(logits, direction='DESCENDING')
        probs_sort = tf.nn.softmax(logits_sort)
        probs_sums = tf.cumsum(probs_sort, axis=1, exclusive=True)
        logits_masked = tf.where(probs_sums < p, logits_sort, tf.ones_like(logits_sort)*1000) # [batchsize, vocab]
        min_logits = tf.reduce_min(logits_masked, axis=1, keepdims=True) # [batchsize, 1]
        return tf.where(
            logits < min_logits,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )


def sample_sequence(
    *,
    hparams,
    length,
    start_token=None,
    target_tokens=None,
    target_bias=None,
    end_tokens=[],
    eval_tokens=None,
    batch_size=None,
    context=None,
    temperature=1,
    top_k=0,
    top_p=0.0,
    gpu_layers=20,
):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = tf.fill([batch_size, 1], start_token)

    if target_tokens is None:
        target_tokens = tf.constant([], dtype=tf.dtypes.int64)
    if target_bias is None:
        target_bias = tf.constant([], dtype=tf.dtypes.float32)

    def step(hparams, tokens, past=None):
        lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE, gpu_layers=gpu_layers)

        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        return {
            'logits': logits,
            'presents': presents,
        }

    with tf.name_scope('sample_sequence'):
        # Don't feed the last context token -- leave that to the loop below
        # TODO: Would be slightly faster if we called step on the entire context,
        # rather than leaving the last token transformer calculation to the while loop.
        context_output = step(hparams, context[:, :-1])

        def body(past, prev, output, evaluate, evaluation):
            next_outputs = step(hparams, prev[:, tf.newaxis], past=past)
            logits = next_outputs['logits'][:, -1, :]  / tf.to_float(temperature)

            # If specified, increase the probability of each target word by its corresponding target_bias
            logits *= tf.reshape(
                tf.sparse.to_dense(
                    tf.SparseTensor(
                        tf.reshape(target_tokens, [-1, 1]),
                        target_bias,
                        [hparams.n_vocab],
                    ),
                    default_value=1,
                ),
                shape=[1, -1],
            )

            # Restrict the logits to either the top_p or top_k tokens
            if top_p > 0.0:
                logits = top_p_logits(logits, p=top_p)
            else:
                logits = top_k_logits(logits, k=top_k)

            if eval_tokens is None:
                next_word = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
                next_evaluation = [[]]
            else:
                next_word = [[evaluate[0]]]
                # I feel like this is wrong, and will likely break when a batch_size > 1 is used
                next_evaluation = [tf.gather(logits, evaluate[0], axis=1)]

            return [
                tf.concat([past, next_outputs['presents']], axis=-2),
                tf.squeeze(next_word, axis=[1]),
                tf.concat([output, next_word], axis=1),
                evaluate[1:],
                tf.concat([evaluation, next_evaluation], axis=1),
            ]

        def cond(past, prev, output, evaluate, evaluation):
            prev_shaped = tf.reshape(prev, shape=[batch_size, 1])
            end_tokens_shaped = tf.constant(end_tokens, dtype=tf.int32, shape=[1, len(end_tokens)])
            end_token_not_seen = tf.math.logical_not(tf.math.reduce_any(tf.math.equal(prev_shaped, end_tokens_shaped)))
            not_end_of_evaluation = True if eval_tokens is None else tf.greater(tf.shape(evaluate)[0], 0)
            return tf.logical_and(not_end_of_evaluation, end_token_not_seen)

        _, _, output, _, evaluation = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=length,
            loop_vars=[
                context_output['presents'],
                context[:, -1],
                context,
                tf.constant([]) if eval_tokens is None else eval_tokens,
                tf.constant([], shape=[batch_size, 0]),
            ],
            shape_invariants=[
                tf.TensorShape(model.past_shape(hparams=hparams, batch_size=batch_size)),
                tf.TensorShape([batch_size]),
                tf.TensorShape([batch_size, None]),
                tf.TensorShape([None]),
                tf.TensorShape([batch_size, None]),
            ],
            back_prop=False,
        )

        if eval_tokens is not None:
            return evaluation
        else:
            return output