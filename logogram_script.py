from svgpathtools import parse_path
from svgpathtools import Path
import svgpathtools
from enum import Enum
import random
import svgwrite
import math
import re
import copy
import cmath
from typing import Tuple, List
from glyph_builder import Character
from glyph_builder import Glyph
from glyph_builder import SvgObject
from glyph_builder import Composition

class LogogramDrawer:

    RADICALS_DEFS = {
        'guy_r': Character(
                Glyph(
                    'M 9 14 L 9 3 M 5 5 C 8 4 8 4 9 3 C 10 2 11 1 11 -1'
                ),
                simplified_glyphs=[
                    Glyph(
                        'M 24 -67 C 37 -77 43 -86 48 -98 C 53 -86 59 -78 71 -67'
                    ),
                ]
            ),
        'moon_r': Character(
                Glyph(
                    'M 28 3 C 28 0 26 -2 23 -2 L 0 -2 L 0 20 C 0 23 2 25 4 25 L 6 25',
                    is_hollow=True
                ),
                simplified_glyphs=[
                    Glyph(
                        'M 32 -3 C -10 -17 -10 50 32 39 C 4 39 4 -3 32 -3'
                    ),
                ]
            ),
        'out_r': Character(
                Glyph(
                    'M 69 -94 L 69 -136 M 78 -104 L 43 -104 C 86 -140 39 -142 46 -125'
                )
            ),
        'male_r': Character(
                Glyph(
                    'M 0 0 C 20 0 20 30 0 30 C -20 30 -20 0 0 0 M 12 6 L 27 -9 C 25 -8 22 -8 16 -9 M 27 -9 C 26 -7 26 -4 27 2'
                )
            ),
        'fmale_r': Character(
                Glyph(
                    'M 0 0 C 20 0 20 30 0 30 C -20 30 -20 0 0 0 M 0 30 L 0 45 M -8 39 L 8 39',
                    margin=[0.05,0.15,0.05,0.15]
                )
            ),
        'home_r': Character(
                Glyph(
                    'M 22 36 C 26 24 25 11 25 -2 L 67 -2',
                    padding=[0.05,0.0,0.0,0.1],
                    is_hollow=True
                ),
            ),
        'time_r': Character(
                Glyph(
                    'M 11 4 L 11 33 M 11 9 C 17 5 34 2 34 9 C 34 12 33.3333 23 33 31 C 33 34 36 34 36 31 M 9 6 L 14 6',
                    is_hollow=True,
                    padding=[0.13,0.13,0.0,0.1]
                ),
                simplified_glyphs=[
                    Glyph(
                        'M 12 3 L 12 44 M 12 9 C 19 2 35 1 37 12 C 40 25 29 29 30 40 C 32 52 44 37 35 35 C 27 34 29 43 17 38 M 8 6 L 17 4',
                        margin=[0.05,0.2,0.05,0.15]
                    ),
                ]
            ),
        'omega_r': Character(
                Glyph(
                    'M 15 -13 L 29 -13 C 14 -20 15 -42 35 -42 C 55 -42 55 -20 40 -13 L 54 -13'
                ),
                simplified_glyphs=[
                    Glyph(
                        'M -5 -16 C -13 -13 -13 0 -5 0 C 0 0 3 -5 0 -13 C -3 -5 0 0 5 0 C 13 0 13 -13 5 -16',
                    ),
                ]
            ),
        'alpha_r': Character(
                Glyph(
                    'M 40 -54 C 39 -43 27 -21 18 -21 C 5 -21 5 -55 24 -55 C 34 -55 32 -37 38 -24 C 40 -19 44 -25 44 -27'
                )
            ),
        'pi_r': Character(
                Glyph(
                    'M 6 -13 L 40 -13 M 6 23 L 40 23 M 9 -13 L 9 23 M 37 -13 L 37 23',
                    is_hollow=True,
                    padding=[0.05,0.1,0.05,0.1],
                ),
                simplified_glyphs=[
                    Glyph(
                        'M 15 -13 L 33 -13 M 15 23 L 33 23 M 20 -13 L 20 23 M 28 -13 L 28 23'
                    ),
                ]
            ),
        'and_r': Character(
                Glyph(
                    'M 6 13 L 6 -47 M 2 -53 C 6 -51 8 -49 9 -47 M 10 -51 L 59 -51 L 59 10 C 59 13 57 14 54 14 L 51 14',
                    is_hollow=True,
                    padding=[0.1, 0.05, 0.05, 0.1]
                )
            ),
        'air_r': Character(
                Glyph(
                    'M 0 -14 L 8 0 L -8 0 Z M -4 -7 L 4 -7',
                ),
                simplified_glyphs=[
                    Glyph(
                        'M 0 0 L 0 -19 L 3 -16 M 0 -16 L 3 -13',
                    ),
                    Glyph(
                        'M 0 0 L 4 -2 L 8 0 L 12 -2 M 0 2 L 4 0 L 8 2 L 12 0',
                    ),
                ]
            ),
        'go_r': Character(
                Glyph(
                    'M 0 -23 L 2 -11 L 0 0 L 23 0 M 0 -17 L 1 -11 L 0 -5',
                    is_hollow=True,
                    padding=[0.0,0.0,0.05,0.1],
                ),
                simplified_glyphs=[
                    Glyph(
                        'M 0 0 L 0 -19 L 4 -15 L 1 -8 L 4 -1',
                    ),
                ]
            ),
        'birth_r': Character(
                Glyph(
                    'M 0 -30 L 0 0 L 7 -8 L 0 -15 L 6 -23 Z',
                    is_hollow=True,
                    padding=[0.1, 0.05, 0.05, 0.1]
                )
            ),
        'fire_r': Character(
                Glyph(
                    'M 0 0 L 2 -10 L 0 -19 L 21 -19',
                    is_hollow=True,
                    padding=[0.05,0.0,0.0,0.1],
                ),
                simplified_glyphs=[
                    Glyph(
                        'M 0 0 L 11 -20 L 22 0 Z M 2 -8 L 6 -15 M 20 -8 L 16 -15',
                    ),
                    Glyph(
                        'M 9 -26 L 11 -16 L 9 -5 M 10 -16 L 8 -22 M 10 -13 L 8 -16',
                    ),
                    Glyph(
                        'M 5 -19 L 2 -21 M 10 -19 L 8 -22 M 14 -19 L 18 -21',
                    ),
                ]
            ),
        'tree_r': Character(
                Glyph(
                    'M 0 -30 L 0 -2 M -13 -25 L 13 -25 M 0 -25 C -3 -15 -4 -13 -12 -6 M 0 -25 C 3 -15 4 -13 12 -6',
                ),
                simplified_glyphs=[
                    Glyph(
                        'M 0 -30 L 0 -2 M -5 -25 L 5 -25 M -3 -24 L -3 -4 M 3 -24 L 3 -4',
                    ),
                ]
            ),
        'stream_r': Character(
                Glyph(
                    'M 12 -19 C 12 -5 12 -5 8 0 M 18 -18 L 18 -2 M 24 -19 L 24 0',
                ),
                simplified_glyphs=[
                    Glyph(
                        'M 11 -19 L 9 -11 L 11 -3 M 13 -19 L 11 -11 L 13 -3',
                    ),
                    Glyph(
                        'M 11 -18 L 8 -16 L 11 -14 M 14 -18 L 11 -16 L 14 -14 M 17 -18 L 14 -16 L 17 -14',
                    ),
                ]
            )
    }

    WORD_COMPONENTS = None

    SIMPLIFICATIONS = {
        ("radical1", "radical2", "radical3"): "combination",
    }


    def __init__(self):
        self.load_vocab()

    def construct_word_tree(self, word):

        # first get the components
        components = []
        if word in self.WORD_COMPONENTS:
            components = self.WORD_COMPONENTS[word]
        elif word in self.RADICALS_DEFS:
            components = [word]
        else:
            print(f"WARNING: '{word}' not a word or radical")

        # now preform any component or radical combination simplifications
        components = self.simplify_components(components)

        trees = []
        # for each component check first if it is a radical
        for component in components:
            if component in self.RADICALS_DEFS:
                trees.append(Composition(leaf_char=self.RADICALS_DEFS[component]))
            elif component in self.WORD_COMPONENTS:
                trees.append(self.construct_word_tree(component))
            else:
                print(f"WARNING: '{component}' not a radical or word")

        l = len(trees)
        while l > 1:
            idx = l - 2
            comp1 = trees[idx]
            comp2 = trees.pop(idx + 1)
            newComp = Composition(
                sub_comp1=comp1,
                sub_comp2=comp2
            )

            trees[idx] = newComp
            l = len(trees)

        trees[0].calc_constructions(True)

        return trees[0]

    def simplify_components(self, components):
        changed = True
        while changed:
            changed = False
            i = 0
            new_components = []

            while i < len(components):
                best_match = None
                best_key = None

                # try all keys, find the longest one that matches here
                for key in self.SIMPLIFICATIONS.keys():
                    klen = len(key)
                    if i + klen <= len(components) and tuple(components[i:i + klen]) == key:
                        if not best_match or klen > len(best_key):
                            best_match = self.SIMPLIFICATIONS[key]
                            best_key = key

                if best_match:
                    new_components.append(best_match)
                    i += len(best_key)
                    changed = True
                else:
                    new_components.append(components[i])
                    i += 1

            components = new_components

        return components

    def get_radicals(self, path):
        with open(path, 'r') as file:
            lines = file.readlines()
            lines = [x.lower().strip() for x in lines]
            return lines

    def get_word_compositions(self, path):
        with open(path, 'r') as file:
            lines = file.readlines()
            pairs = {}
            for line in lines:
                line = line.split('#', 1)[0].strip(" \n\t")
                if line.startswith("#") or ',' not in line:
                    continue

                data = line.split(',')
                if len(data) < 4 or len(data[0].split(' ')) != 1:
                    continue

                word = data[0]
                rads = data[3].split(' ')
                pairs[word] = rads
            return pairs

    def load_vocab(self):

        rad_path = "radicals.csv"
        vocab_path = "vocab.csv"

        radicals = self.get_radicals(rad_path)

        # first check if we have definitions for all the radicals
        missing_radicals = []
        for r in radicals:
            if r not in self.RADICALS_DEFS and r not in missing_radicals:
                missing_radicals.append(r)

        if len(missing_radicals) > 0:
            print(f"Missing radical svg definitions: {missing_radicals}")

        # laod in the word->radicals set
        self.WORD_COMPONENTS = self.get_word_compositions(vocab_path)
    
    def draw_to_svg_objs(self, sentence, size=200, stroke_width=5):
        words = sentence.split(' ')

        word_trees = []
        for w in words:
            t = self.construct_word_tree(w)
            word_trees.append(t)

        char_gap = size / 8
        dims = ((size * len(words)) + ((len(words) - 1) * char_gap), size)

        dwg = svgwrite.Drawing()
        x = 0
        svgs = []
        for i in range(len(word_trees)):
            t = word_trees[i]
            svg_obj = t.create_svg_obj(dwg, size, stroke_width)
            svgs.append(svg_obj)

        return svgs

    def draw_to_svg_obj(self, sentence, size=200, stroke_width=5):
        
        svgs = self.draw_to_svg_objs(sentance, size, stroke_width)

        dwg = svgwrite.Drawing()
        sentence_group = dwg.g()
        char_gap = size / 8
        x = 0
        for i in range(len(svgs)):
            svgs[i].set_xy(x, 0)
            svgs[i].draw_to_group(sentence_group)
            x += size + char_gap

        dims = ((size * len(svgs)) + ((len(svgs) - 1) * char_gap), size)
        return SvgObject(sentence_group, dims[0], dims[1])

sentance = "front back"
drawer = LogogramDrawer()

s = 400
dwg = svgwrite.Drawing('out.svg')
dwg.add(dwg.rect(insert=(0, 0), size=("100%", "100%"), fill="white"))
s = drawer.draw_to_svg_obj(sentance)
dwg['width'] = s.width
dwg['height'] = s.height
s.draw_to_canvas(dwg)
dwg.save()