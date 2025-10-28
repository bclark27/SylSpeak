from svgpathtools import parse_path
from svgpathtools import Path
from enum import Enum
import random
import svgwrite
import math

class CompositionOp(Enum):
    VERT=1
    HORZ=2
    IN=3
    NONE=4

# --- Glyph definition ---
class Glyph:
    def __init__(self, 
                path_str, 
                is_hollow=False,
                padding=[0.1, 0.1, 0.1, 0.1], # top right bot left (interior space)
                margin=[0.05, 0.05, 0.05, 0.05], # top right bot left (outside)
                translate=[0, 0], # x y (translate based on fraction of the width of the final area)
                ):
        self.is_hollow = is_hollow
        self.padding = padding
        self.margin = margin
        self.translate = translate

        self.path = Glyph.normalize_path(path_str)
        minx, maxx, miny, maxy = self.path.bbox()
        self.width = maxx - minx
        self.height = maxy - miny

        top = self.margin[0] * self.height
        right = self.margin[1] * self.width
        bot = self.margin[2] * self.height
        left = self.margin[3] * self.width

        self.width += right + left
        self.height += top + bot

        print(self.width, self.height)

    def normalize_path(path):

        if path is None:
            return None

        p = parse_path(path)
        
        minx, maxx, miny, maxy = p.bbox()
        width = maxx - minx
        height = maxy - miny

        cx = width / 2
        cy = height / 2
        p = (
            p
            .translated(-complex(cx, cy))
            .scaled(-1, -1)
            .translated(complex(cx, cy))
        )

        p = p.scaled(-1, 1)

        minx, maxx, miny, maxy= p.bbox()
        p = p.translated(-complex(minx, miny))
        return p

    def draw(self, dwg, draw_w, draw_h, x, y, stroke):

        p = self.path
   
        minx, maxx, miny, maxy = p.bbox()
        path_width = maxx - minx
        path_height = maxy - miny

        scale_x = draw_w / path_width
        scale_y = draw_h / path_height
        
        # scale to fit the draw canvas size
        p = p.scaled(scale_x, scale_y)
        # flip on the y axis
        p = p.scaled(1, -1)
        # scale it down to fit the margins
        p = p.scaled(1 - (self.margin[1] + self.margin[3]), 1 - (self.margin[0] + self.margin[2]))
        
        # Translate to top-left of allocated box
        p = p.translated(complex(x, y + draw_h))

        # translate the path to go into the middle of the margin
        p = p.translated(complex(draw_w * self.margin[3], -draw_h * self.margin[2]))
        
        # translate optional amount outside the bounds
        p = p.translated(complex(draw_w * self.translate[0], draw_h * self.translate[1]))
        
        dwg.add(dwg.path(d=p.d(), stroke='black', fill='none', stroke_width=stroke))


class Character:
    def __init__(self, main_glyph, simplified_glyphs=[]):
        self.main_glyph = main_glyph
        self.simplified_glyphs = simplified_glyphs

    def draw(self, dwg, draw_w, draw_h, x, y, stroke, can_simplify):
        if can_simplify:
            best_fit = self.get_best_glyph((draw_h, draw_w))
            best_fit.draw(dwg, draw_w, draw_h, x, y, stroke)
        else:
            self.main_glyph.draw(dwg, draw_w, draw_h, x, y, stroke)

    def closest_aspect_ratio_index(glyphs, ref_hw):
        """
        Returns the index in hw_list whose aspect ratio (h/w)
        is closest to that of ref_hw (reference height/width tuple).
        """
        ref_h, ref_w = ref_hw
        ref_ratio = ref_h / ref_w

        closest_idx = None
        closest_diff = float('inf')

        for i, g in enumerate(glyphs):
            if g is None:
                continue
            w = g.width
            h = g.height
            ratio = h / w
            diff = abs(ratio - ref_ratio)
            if diff < closest_diff:
                closest_diff = diff
                closest_idx = i

        return closest_idx

    def get_best_glyph(self, ref_hw):
        options = [self.main_glyph] + self.simplified_glyphs
        return options[Character.closest_aspect_ratio_index(options, ref_hw)]


class Composition:
    def __init__(self, op=None, sub_comp1=None, sub_comp2=None, leaf_char=None):
        self.op = op
        self.sub_comp1 = sub_comp1
        self.sub_comp2 = sub_comp2
        self.leaf_char = leaf_char
        self.sub_comp1_percent = 0.5
        self.computed = False

    def is_leaf(self):
        return self.leaf_char is not None

    def is_hollow(self):
        return self.is_leaf() and self.leaf_char.main_glyph.is_hollow

    def get_composition_side_ratio(self):
        if self.is_leaf():
            return (self.leaf_char.main_glyph.width, self.leaf_char.main_glyph.height)

        # Recursively get child sizes
        w1, h1 = self.sub_comp1.get_composition_side_ratio()
        w2, h2 = self.sub_comp2.get_composition_side_ratio()

        if self.op == CompositionOp.VERT:
            total_height = h1 + h2
            self.sub_comp1_percent = h1 / total_height if total_height != 0 else 0.5
            w = max(w1, w2)
            h = total_height
        elif self.op == CompositionOp.HORZ:
            total_width = w1 + w2
            self.sub_comp1_percent = w1 / total_width if total_width != 0 else 0.5
            w = total_width
            h = max(h1, h2)
        elif self.op == CompositionOp.IN:
            w = max(w1, w2)
            h = max(h1, h2)
            self.sub_comp1_percent = 0.5
        elif self.op == CompositionOp.NONE:
            w, h = w1, h1
            self.sub_comp1_percent = 1.0
        else:
            raise ValueError(f"Unknown composition operation: {self.op}")

        return (w, h)


    def calc_constructions(self, donot_recompute=False):

        if self.computed and donot_recompute:
            return

        self.computed = True

        if self.is_leaf():
            self.op = CompositionOp.NONE
            self.sub_comp1_percent = 1.0
            return

        # Recursively compute child constructions
        self.sub_comp1.calc_constructions()
        self.sub_comp2.calc_constructions()

        # Hollow nesting check
        if self.sub_comp1.is_hollow():# and self.sub_comp2.op != CompositionOp.IN:
            self.op = CompositionOp.IN
            self.sub_comp1_percent = 0.5
            return

        # --- get child sizes ---
        w1, h1 = self.sub_comp1.get_composition_side_ratio()
        w2, h2 = self.sub_comp2.get_composition_side_ratio()

        # --- vertical stacking distortion ---
        vert_total_height = h1 + h2
        vert_sub1_percent = h1 / vert_total_height if vert_total_height != 0 else 0.5
        vert_sub2_percent = h2 / vert_total_height if vert_total_height != 0 else 0.5
        vert_scale_x = 1 / max(w1, w2)
        vert_dist1 = max(vert_scale_x, vert_sub1_percent) / min(vert_scale_x, vert_sub1_percent)
        vert_dist2 = max(vert_scale_x, vert_sub2_percent) / min(vert_scale_x, vert_sub2_percent)
        vert_max_distortion = max(vert_dist1, vert_dist2)

        # --- horizontal stacking distortion ---
        horz_total_width = w1 + w2
        horz_sub1_percent = w1 / horz_total_width if horz_total_width != 0 else 0.5
        horz_sub2_percent = w2 / horz_total_width if horz_total_width != 0 else 0.5
        horz_scale_y = 1 / max(h1, h2)
        horz_dist1 = max(horz_scale_y, horz_sub1_percent) / min(horz_scale_y, horz_sub1_percent)
        horz_dist2 = max(horz_scale_y, horz_sub2_percent) / min(horz_scale_y, horz_sub2_percent)
        horz_max_distortion = max(horz_dist1, horz_dist2)

        # --- pick stacking with less distortion ---
        if vert_max_distortion >= horz_max_distortion:
            self.op = CompositionOp.VERT
            self.sub_comp1_percent = vert_sub1_percent
        else:
            self.op = CompositionOp.HORZ
            self.sub_comp1_percent = horz_sub1_percent

    def draw_svg(self, dwg, pos_x, pos_y, size=400, stroke=5):
        """
        Draws this composition as an independent SVG.
        - size: final image size in pixels (square)
        - stroke: stroke width in pixels
        """
        def draw_node(comp, x, y, w, h, char_is_empty=True):
            """
            Recursive helper to draw a composition node.
            - comp: current Composition
            - x, y: top-left coordinates
            - w, h: width and height to draw this node
            """
            # Add a tiny gap for stacked compositions

            if comp.is_leaf():
                comp.leaf_char.draw(dwg, w, h, x, y, stroke, char_is_empty)

            elif comp.op == CompositionOp.VERT:
                # Vertical stack: divide height according to sub_comp1_percent
                h1 = h * comp.sub_comp1_percent
                h2 = h - h1
                draw_node(comp.sub_comp1, x, y, w, h1)
                draw_node(comp.sub_comp2, x, y + h1, w, h2)
            elif comp.op == CompositionOp.HORZ:
                # Horizontal stack: divide width according to sub_comp1_percent
                w1 = w * comp.sub_comp1_percent
                w2 = w - w1
                draw_node(comp.sub_comp1, x, y, w1, h)
                draw_node(comp.sub_comp2, x + w1, y, w2, h)
            elif comp.op == CompositionOp.IN:
                # IN operation: sub_comp2 is nested inside sub_comp1
                # Apply inside margins from the leaf glyph (assume sub_comp1 is hollow)
                outer = comp.sub_comp1
                inner = comp.sub_comp2
                if outer.is_leaf():
                    padding = outer.leaf_char.main_glyph.padding
                    margin = outer.leaf_char.main_glyph.margin
                else:
                    padding = [0.05,0.05,0.05,0.05]  # default if not leaf
                    margin = [0.05,0.05,0.05,0.05]  # default if not leaf
                
                # Convert margins to pixel space
                outer_w = w * (1 - margin[1] - margin[3])
                inner_x = x + (w * margin[3]) + (outer_w * padding[3])

                outer_h = h * (1 - margin[0] - margin[2])
                inner_y = y + (h * margin[0]) + (outer_h * padding[0])

                inner_w = (w * (1 - margin[1] - margin[3])) * (1 - padding[1] - padding[3])
                inner_h = (h * (1 - margin[0] - margin[2])) * (1 - padding[0] - padding[2])

                draw_node(outer, x, y, w, h, False)
                draw_node(inner, inner_x, inner_y, inner_w, inner_h)
            else:
                raise ValueError(f"Unknown composition operation: {comp.op}")

        # Start recursive drawing
        draw_node(self, pos_x, pos_y, size, size)

CHARACTERS = {
    's': Character(
            Glyph(
            'M 38 -149 L 38 -113 M 33 -145 L 48 -145 M 38 -134 C 51 -155 79 -130 48 -113 C 45 -111 48 -106 53 -108',
            margin=[0.05,0.15,0.05,0.15])
        ),
    'c': Character(
            Glyph(
            'M 34 -124 L 34 -137 C 49 -137 49 -159 34 -159 C 28 -159 24 -155 24 -148 M 39 -131 L 28 -131',
            margin=[0.05,0.2,0.05,0.2])
        ),
    'w': Character(Glyph( 
        'M 20 -34 C 23 -34 22 -47 22 -54 L 40 -54 L 40 -34',
        is_hollow=True,
        padding=[0.1, 0.1, 0.05, 0.2])),
    'g': Character(
            Glyph(
                'M 9 14 L 9 3 M 5 5 C 8 4 8 4 9 3 C 10 2 11 1 11 -1'
            ),
            simplified_glyphs=[
                Glyph(
                    'M 24 -67 C 37 -77 43 -86 48 -98 C 53 -86 59 -78 71 -67',
                ),
            ]
        ),
    'h': Character(
            Glyph(
                'M 20 -143 L 20 -124 M 20 -152 L 39 -152 C 41 -152 43 -150 43 -148 L 43 -124',
                is_hollow=True,
                padding=[0.1,0.1,0,0.1]
            )
        ),
    'd': Character(Glyph(
        'M 20 15 C 17 22 13 26 6 33 M 28 31 C 22 31 16 31 7 32 M 29 33 C 28 31 26 28 25 25 M 14 19 C 12 22 11 24 8 27',
        is_hollow=False)),
    'f': Character(Glyph(
        'M 28 26 C 39 20 53 22 65 26 C 47 10 51 1 54 -3 M 65 26 L 28 -7 M 41 -4 L 31 5',
        is_hollow=False)),
    'sh': Character(
        Glyph(
                'M 6 13 L 6 -47 M 2 -53 C 6 -51 8 -49 9 -47 M 10 -51 L 59 -51 L 59 10 C 59 13 57 14 54 14 L 51 14',
                is_hollow=True,
                padding=[0.1, 0.05, 0.05, 0.1]
            )
        ),
    'z': Character(
            Glyph(
                'M 40 -149 L 40 -103 C 33 -103 33 -93 40 -93 C 47 -93 47 -103 40 -103 M 27 -127 L 54 -127 M 20 -149 C 30 -127 30 -127 20 -106 M 61 -149 C 51 -127 51 -127 61 -106'
            )
        ),
    'j': Character(
            Glyph(
                'M 69 -94 L 69 -136 M 78 -104 L 43 -104 C 86 -140 39 -142 46 -125'
            )
        ),
    'n': Character(
            Glyph(
                'M 11 4 L 11 33 M 11 7 C 16 4 33 3 34 7 C 36 12 33.3333 23 33 31 C 33 34 36 34 36 31',
                is_hollow=True,
                padding=[0.1,0.15,0.02,0.07]
            ),
            simplified_glyphs=[
                Glyph(
                    'M 12 3 L 12 46 M 12 9 C 19 2 31 1 33 12 C 35 26 29 29 31 43 C 34 52 44 39 33 38 C 28 38 30 45 23 42'
                ),
                Glyph(
                    'M 12 3 L 12 19 M 12 8 C 19 2 39 -1 43 4 C 47 9 42 10 43 15 C 46 25 54 11 45 12 C 41 13 43 15 38 17'
                ),
            ]
        ),
    'p': Character(
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
    'm': Character(
            Glyph(
                'M 11 11 C 13 8 16 10 16 16 L 16 38 M 15 12 C 20 8 24 10 24 16 L 24 38 M 23 12 C 28 8 32 10 32 16 C 32 38 33 36 37 38 M 32 17 C 34 15 37 15 37 18 C 37 27 35 34 29 39',
                margin=[0.05,0.1,0.05,0.1]
            )
        ),
    'o': Character(
            Glyph(
                'M 15 -13 L 29 -13 C 14 -20 15 -42 35 -42 C 55 -42 55 -20 40 -13 L 54 -13'
            )
        )
}

SUB_CHARACTERS = {
    ''
}

def segment_word(word, sequences):
    """
    Splits 'word' into a list of substrings found in 'sequences',
    using greedy longest-match from left to right.
    Characters not in any sequence are skipped.
    """
    result = []
    i = 0
    while i < len(word):
        best_match = None
        # Try longest possible substring from this position
        for j in range(len(word), i, -1):
            segment = word[i:j]
            if segment in sequences:
                best_match = segment
                break
        if best_match:
            result.append(best_match)
            i += len(best_match)
        else:
            # Skip one char if no match
            result.append(word[i])
            i += 1
    return result

#TODO: adda word finder, which searches for words and then tries to keep that character pre calculated and consistent


def create_tree(word):
    segs = segment_word(word, CHARACTERS)
    comps = []
    for seg in segs:
        comps.append(Composition(leaf_char=CHARACTERS[seg]))

    l = len(comps)
    while l > 1:
        idx = l - 2
        comp1 = comps[idx]
        comp2 = comps.pop(idx + 1)
        newComp = Composition(
            sub_comp1=comp1,
            sub_comp2=comp2
        )

        comps[idx] = newComp
        l = len(comps)

    return comps[0]

def get_sub_character_tree(word):
    segs = segment_word(word, SUB_CHARACTERS)
    sub_char_trees = []
    for seg in segs:
        t = create_tree(seg)
        t.calc_constructions()
        sub_char_trees.append(t)

    l = len(sub_char_trees)
    while l > 1:
        idx = l - 2
        comp1 = sub_char_trees[idx]
        comp2 = sub_char_trees.pop(idx + 1)
        newComp = Composition(
            sub_comp1=comp1,
            sub_comp2=comp2
        )

        sub_char_trees[idx] = newComp
        l = len(sub_char_trees)

    sub_char_trees[0].calc_constructions(True)

    return sub_char_trees[0]
    

def draw_sentence(sentence, filename, size=400, stroke=5):
    words = sentence.split(' ')
    
    word_trees = []
    for w in words:
        t = create_tree(w)
        t.calc_constructions(True)
        word_trees.append(t)

    char_gap = size / 8
    dims = ((size * len(words)) + ((len(words) - 1) * char_gap), size)
    dwg = svgwrite.Drawing(filename, size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))

    x = 0
    for i in range(len(word_trees)):
        t = word_trees[i]
        t.draw_svg(dwg, x, 0, size, stroke)
        x += size + char_gap

    dwg.save()

    
draw_sentence('npjmshz', 'out.svg', 200, 5)
