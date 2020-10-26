#!/usr/bin/env python3

from typing import List
from typing import Set
from typing import Text

import dataclasses
import json
import math
import os
import random
import re

import fire
import numpy as np
import nvidia_smi
import tensorflow as tf

import model
import sample
import encoder

def storyteller(
    model_name='1558M',
    seed=None,
    temperature=1,
    top_k=40,
    top_p=0.0,
    eval_user=False,
    gpu_layers=None,
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
    :eval_user=False : Whether to evaluate the quality of the user's input with
     respect to the model's predictions.
    :gpu_layers=None : The amount of transformer layers to assign to the GPU.
     By default a number is chosen based on available GPU memory and heuristics.
    """
    gpu_mem_before = get_gpu_memory()
    if gpu_layers is None:
        gpu_layers = estimate_gpu_layers(gpu_mem_before)

    with StoryServer(
        model_name=model_name,
        seed=seed,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eval_user=eval_user,
        gpu_layers=gpu_layers,
    ) as story_server:

        def get_target_token(prompt: Text):
            while True:
                try:
                    return story_server.encode_word(input(prompt))
                except ValueError:
                    print('Target word not in vocab')

        gpu_mem_after = get_gpu_memory()
        print('Total GPU memory:', gpu_mem_after.total)
        print(f'Free GPU memory: {gpu_mem_after.free} ({gpu_mem_after.free/gpu_mem_after.total*100:.2g}%)')
        print(f'Used GPU memory: {gpu_mem_after.used} ({gpu_mem_after.used/gpu_mem_after.total*100:.2g}%)')
        print(f'Used GPU memory by model: {gpu_mem_after.used - gpu_mem_before.used} ({(gpu_mem_after.used - gpu_mem_before.used)/gpu_mem_after.total*100:.2g}%)')

        while True:
            # Setup
            team_num = int(input(f"How many teams? >>> "))
            teams = [Team() for _ in range(team_num)]
            team_turn = random.randrange(team_num)

            for t, team in enumerate(teams):
                name = None
                while name != '':
                    if name is not None:
                        team.players.append(Player(name))
                    name = input(f"Enter the name of Player {len(team.players)+1} for Team {t+1} >>> ")

            target_word_num = int(input(f"How many target words for each team? >>> "))
            for t, team in enumerate(teams):
                for target in range(target_word_num):
                    target_token = get_target_token(f"Select target word {target+1} for Team {t+1} >>> ")
                    team.target_tokens.append(target_token)

            # Start the story
            story = 'A short story\nBy John Smith\n\nIt began like this. '
            text = story_server.expand_story(story)
            story += text
            print(text)

            def get_contained_target_tokens(sentence: Text, team: Team):
                tokens = []
                for target_token in team.unseen_tokens():
                    target_word = story_server.decode_token(target_token)
                    if re.search(r'\b' + target_word, sentence):
                        tokens.append(target_token)
                return tokens

            def get_user_sentence(team: Team):
                player = team.current_player()
                while True:
                    user_sentence = input(f"{player.name} >>> ")

                    if not user_sentence:
                        print('Prompt should not be empty!')
                        continue

                    contained_target_tokens = get_contained_target_tokens(user_sentence, team)
                    if contained_target_tokens:
                        print(
                            'Your sentence cannot contain your team\'s target tokens: ',
                            [story_server.decode_token(target_token) for target_token in contained_target_tokens],
                        )
                        continue

                    if user_sentence == 'quit':
                        return None

                    # Add a full stop if it wasn't included
                    if not any(user_sentence.rstrip().endswith(end_string) for end_string in story_server.end_strings):
                        user_sentence = user_sentence.rstrip() + '. '
                    # Ensure that the sentence doesn't immediately terminate
                    if not user_sentence.endswith(' '):
                        user_sentence += ' '

                    return user_sentence

            winners = []
            while not winners:
                team_turn = (team_turn + 1) % len(teams)
                current_team = teams[team_turn]
                current_team.next_turn()

                user_sentence = get_user_sentence(current_team)
                if user_sentence is None:
                    break

                if eval_user:
                    results = story_server.evaluate_user(story, user_sentence)
                    scores = [result[1] for result in results]
                    print(results)
                    print('Average:', sum(scores)/len(scores))
                    print('Min:', min(scores))

                story += ' ' + user_sentence
                text = story_server.expand_story(
                    story,
                    sorted(target_token for team in teams for target_token in team.unseen_tokens()),
                    1.5,
                )
                story += text
                print(text)

                for t, team in enumerate(teams):
                    contained_target_tokens = get_contained_target_tokens(text, team)
                    if contained_target_tokens:
                        print(f"Team {t+1} scored:", [story_server.decode_token(target_token) for target_token in contained_target_tokens])
                        team.seen_tokens |= set(contained_target_tokens)

                for t, team in enumerate(teams):
                    if not team.unseen_tokens():
                        winners.append(t)

            for winner in winners:
                print("~~~ The End ~~~")
                print(f"Team {winner+1} wins!", [player.name for player in teams[winner].players])


@dataclasses.dataclass
class Player:
    name: Text


@dataclasses.dataclass
class Team:
    players: List[Player] = dataclasses.field(default_factory=list)
    target_tokens: List[int] = dataclasses.field(default_factory=list)
    seen_tokens: Set[int] = dataclasses.field(default_factory=set)
    turn: int = 0

    def current_player(self):
        return self.players[self.turn]

    def next_turn(self):
        self.turn = (self.turn + 1) % len(self.players)

    def unseen_tokens(self):
        return set(self.target_tokens) - self.seen_tokens


class StoryServer:

    def __init__(self,
        model_name='1558M',
        seed=None,
        temperature=1,
        top_k=40,
        top_p=0.0,
        eval_user=False,
        gpu_layers=None,
    ):
        self.model_name = model_name
        self.seed = seed
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.gpu_layers = gpu_layers

        self.enc = encoder.get_encoder(model_name)
        self.hparams = self._get_hparams()
        self.end_strings = self._get_end_strings()
        self.end_tokens = [ self.enc.encode(end_string)[0] for end_string in self.end_strings ]

        self.sess = tf.Session(graph=tf.Graph(), config=self._get_config())

    def _get_hparams(self):
        hparams = model.default_hparams()
        with open(os.path.join('models', self.model_name, 'hparams.json')) as f:
            hparams.override_from_dict(json.load(f))
        return hparams

    def _get_config(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.force_gpu_compatible = True
        return config

    def _get_end_strings(self):
        # Hacky end-of-sentence detection
        adjacent_punctuation = list('()"\'')
        ending_punctuation = list('.!?…')
        end_strings = ending_punctuation + [
            adjacent + ending
            for adjacent in adjacent_punctuation
            for ending in ending_punctuation
        ] + [
            ending + adjacent
            for adjacent in adjacent_punctuation
            for ending in ending_punctuation
        ]
        return end_strings

    def __enter__(self):
        self.sess.__enter__()

        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        self.context = tf.placeholder(tf.int32, [1, None])
        self.target_tokens = tf.placeholder(tf.int64, [None])
        self.target_bias = tf.placeholder(tf.float32, [None])
        self.eval_tokens = tf.placeholder(tf.int32, [None])

        self.output = sample.sample_sequence(
            hparams=self.hparams, length=100,
            context=self.context,
            end_tokens=self.end_tokens,
            target_tokens=self.target_tokens,
            target_bias=self.target_bias,
            batch_size=1,
            temperature=self.temperature, top_k=self.top_k, top_p=self.top_p,
            gpu_layers=self.gpu_layers,
        )
        self.evaluation = sample.sample_sequence(
            hparams=self.hparams, length=100,
            context=self.context,
            eval_tokens=self.eval_tokens,
            batch_size=1,
            temperature=self.temperature, top_k=0,
            gpu_layers=self.gpu_layers,
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', self.model_name))
        saver.restore(self.sess, ckpt)

        return self

    def __exit__(self, *args):
        return self.sess.__exit__(*args)

    def encode_word(self, word):
        encoded = self.enc.encode(' ' + word.lstrip())
        if len(encoded) != 1:
            raise ValueError('Target word not in vocab')
        return encoded[0]

    def decode_token(self, token):
        decoded = self.enc.decode([token])
        return decoded.strip()

    def expand_story(self, story, target_tokens=[], bias=1.0):
        run_options = tf.compat.v1.RunOptions(timeout_in_ms=30_000)
        expansion = ''

        for _ in range(1):  # TODO: Figure out a way of stopping huge run on sentences, then increase this
            context_tokens = self.enc.encode(story + expansion)
            out = self.sess.run(self.output, options=run_options, feed_dict={
                self.context: [context_tokens],
                self.target_tokens: target_tokens,
                self.target_bias: [bias] * len(target_tokens),
            })[:, len(context_tokens):]

            text = self.enc.decode(out[0])
            expansion += text

            if any(expansion.endswith(end_string) for end_string in self.end_strings):
                break

        return expansion

    def evaluate_user(self, story, user_sentence):
        run_options = tf.compat.v1.RunOptions(timeout_in_ms=30_000)

        context_tokens = self.enc.encode(story)
        evaluation_tokens = self.enc.encode(' ' + user_sentence)

        eval_result = self.sess.run(self.evaluation, options=run_options, feed_dict={
            self.context: [context_tokens],
            self.eval_tokens: evaluation_tokens,
        })[0]

        return list(
            (self.enc.decode([token]), result)
            for token, result in zip(evaluation_tokens, eval_result)
        )


def get_gpu_memory():
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    nvidia_smi.nvmlShutdown()
    return info


BASE_MEM_USAGE = 726_835_200
LAYER_MEM_USAGE = 200_000_000


def estimate_gpu_layers(gpu_memory):
    gpu_layers = math.floor((gpu_memory.free-BASE_MEM_USAGE)/LAYER_MEM_USAGE)
    print(f"Using {gpu_layers} GPU layers")
    return gpu_layers


if __name__ == '__main__':
    fire.Fire(storyteller)
