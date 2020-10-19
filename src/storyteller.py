#!/usr/bin/env python3

import fire
import json
import os
import re
import numpy as np
import tensorflow as tf

import model, sample, encoder

def storyteller(
    model_name='1558M',
    seed=None,
    temperature=1,
    top_k=40,
    top_p=0.0,
):
    """
    Interactively run the model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    :top_p=0.0 : Float value controlling diversity. Implements nucleus sampling,
     overriding top_k if set to a value > 0. A good setting is 0.9.
    """
    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    # Hacky end-of-sentence detection
    adjacent_punctuation = list('()"\'')
    ending_punctuation = list('.!?')
    end_strings = ending_punctuation + [
        adjacent + ending
        for adjacent in adjacent_punctuation
        for ending in ending_punctuation
    ] + [
        ending + adjacent
        for adjacent in adjacent_punctuation
        for ending in ending_punctuation
    ]
    end_tokens = [ enc.encode(end_string)[0] for end_string in end_strings ]

    config = tf.ConfigProto()

    with tf.Session(graph=tf.Graph(), config=config) as sess:

        context = tf.placeholder(tf.int32, [1, None])
        target_token = tf.placeholder(tf.int32, [])
        eval_tokens = tf.placeholder(tf.int32, [None])
        np.random.seed(seed)
        tf.set_random_seed(seed)

        output = sample.sample_sequence(
            hparams=hparams, length=100,
            context=context,
            end_tokens=end_tokens,
            target_token=target_token,
            batch_size=1,
            temperature=temperature, top_k=top_k, top_p=top_p
        )
        evaluation = sample.sample_sequence(
            hparams=hparams, length=100,
            context=context,
            eval_tokens=eval_tokens,
            batch_size=1,
            temperature=temperature, top_k=0,
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)

        while True:
            target_word = input("Select a target word >>> ")
            target_encoded = enc.encode(' ' + target_word.lstrip())
            if len(target_encoded) != 1:
                print('Target word not in vocab')
                continue
            target = target_encoded[0]
            story = ''

            while not any(story.endswith(end_string) for end_string in end_strings):
                context_tokens = enc.encode('A short story\nBy John Smith\n\nIt began like this. ' + story)
                out = sess.run(output, feed_dict={
                    context: [context_tokens],
                    target_token: -1,
                })[:, len(context_tokens):]

                text = enc.decode(out[0])
                print(text, end='')
                story += text
            print()

            while True:
                while True:
                    user_sentence = input(">>> ")
                    if not user_sentence:
                        print('Prompt should not be empty!')
                        continue
                    if re.search(r'\b' + target_word, user_sentence):
                        print('You can\'t use the target word in your own sentence.')
                        continue
                    break

                if user_sentence == 'quit':
                    break

                # Add a full stop if it wasn't included
                if not any(user_sentence.rstrip().endswith(end_string) for end_string in end_strings):
                    user_sentence = user_sentence.rstrip() + '. '
                # Ensure that the sentence doesn't immediately terminate
                if not user_sentence.endswith(' '):
                    user_sentence += ' '

                context_tokens = enc.encode(story)
                evaluation_tokens = enc.encode(' ' + user_sentence)

                eval_result = sess.run(evaluation, feed_dict={
                    context: [context_tokens],
                    eval_tokens: evaluation_tokens,
                })[0]
                for token, result in zip(evaluation_tokens, eval_result):
                    print(enc.decode([token]), result)
                print('Average:', sum(eval_result)/len(eval_result))
                print('Min:', min(eval_result))

                story += ' ' + user_sentence

                while not any(story.endswith(end_string) for end_string in end_strings):
                    context_tokens = enc.encode(story)
                    out = sess.run(output, feed_dict={
                        context: [context_tokens],
                        target_token: target,
                    })[:, len(context_tokens):]

                    text = enc.decode(out[0])
                    print(text, end='')
                    story += text
                print()

                if re.search(r'\b' + target_word, text):
                    break

            print('You win!')

if __name__ == '__main__':
    fire.Fire(storyteller)
