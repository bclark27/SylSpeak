from svgpathtools import parse_path
from svgpathtools import Path
from enum import Enum
import random
import svgwrite
import math
import copy

class CompositionOp(Enum):
    VERT=1
    HORZ=2
    IN=3
    NONE=4

MARG_DEF = 0.03
PADD_DEF = 0.07

class SvgObject:
    def __init__(self, group, width, height, center_x=0, center_y=0):
        self.group = group
        self.width = width
        self.height = height
        self.center_x = center_x
        self.center_y = center_y

        # Absolute transform state
        self.x = 0
        self.y = 0
        self.angle = 0
        self.scale_x = 1
        self.scale_y = 1

        # Transform origin as normalized coordinates inside bounding box
        self.origin_rel_x = 0.5
        self.origin_rel_y = 0.5

    # --- Public setters ---
    def set_xy(self, new_x, new_y):
        self.x = new_x
        self.y = new_y
        self._update_transform()

    def set_rotate(self, angle):
        self.angle = angle
        self._update_transform()

    def set_scale(self, scale_x, scale_y=None):
        self.scale_x = scale_x
        self.scale_y = scale_y if scale_y is not None else scale_x
        self._update_transform()

    def set_origin(self, rel_x, rel_y):
        """Set rotation/scale origin as relative coordinates inside [0,1]x[0,1]"""
        self.origin_rel_x = rel_x
        self.origin_rel_y = rel_y
        self._update_transform()

    # --- Internal helpers ---
    def _update_transform(self):
        transforms = []

        # Compute the absolute origin point (inside local bounding box)
        origin_abs_x = self.width * self.origin_rel_x
        origin_abs_y = self.height * self.origin_rel_y

        # Apply transformations in SVG order: translate → rotate → scale
        if self.x != 0 or self.y != 0:
            transforms.append(f"translate({self.x},{self.y})")

        if self.angle != 0:
            transforms.append(f"rotate({self.angle},{origin_abs_x},{origin_abs_y})")

        if self.scale_x != 1 or self.scale_y != 1:
            transforms.append(f"scale({self.scale_x},{self.scale_y})")

        # Set or remove transform attribute
        if transforms:
            self.group.attribs['transform'] = " ".join(transforms)
        elif 'transform' in self.group.attribs:
            del self.group.attribs['transform']

    # --- Drawing methods ---
    def draw_to_canvas(self, dwg):
        """Draw this object directly into the drawing"""
        self._draw_to_parent(dwg)

    def draw_to_group(self, parent_group):
        """Draw this object into a given parent group (nested)"""
        self._draw_to_parent(parent_group)

    def _draw_to_parent(self, parent):
        """Internal helper for drawing to either canvas or group"""
        cpy = copy.deepcopy(self.group)
        transform_value = self.group.attribs.get('transform')
        if transform_value:
            cpy.attribs['transform'] = transform_value
        parent.add(cpy)


# --- Glyph definition ---
class Glyph:
    def __init__(self, 
                path_str, 
                is_hollow=False,
                padding=[PADD_DEF]*4, # top right bot left (interior space)
                margin=[MARG_DEF]*4, # top right bot left (outside)
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

    def draw(self, dwg, parent, draw_w, draw_h, x, y, stroke):

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
        
        parent.add(dwg.path(
            d=p.d(), 
            stroke='black', 
            fill='none', 
            stroke_width=stroke
        ))


class Character:
    def __init__(self, main_glyph, simplified_glyphs=[]):
        self.main_glyph = main_glyph
        self.simplified_glyphs = simplified_glyphs

    def draw(self, dwg, parent, draw_w, draw_h, x, y, stroke, can_simplify):
        if can_simplify:
            best_fit = self.get_best_filled_glyph((draw_h, draw_w))
            if best_fit is None:
                best_fit = self.get_best_glyph((draw_h, draw_w))
            best_fit.draw(dwg, parent, draw_w, draw_h, x, y, stroke)
        else:
            self.main_glyph.draw(dwg, parent, draw_w, draw_h, x, y, stroke)

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

    def get_best_filled_glyph(self, ref_hw):
        options = list(filter(lambda x: not x.is_hollow, [self.main_glyph] + self.simplified_glyphs))
        best_idx = Character.closest_aspect_ratio_index(options, ref_hw)
        if best_idx is None:
            return None
        return options[best_idx]

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

    def create_svg_obj(self, dwg, size=400, stroke=5):
        """
        Draws this composition as an independent SVG.
        - size: final image size in pixels (square)
        - stroke: stroke width in pixels
        """

        group = dwg.g()

        def draw_node(comp, x, y, w, h, char_is_empty=True):
            """
            Recursive helper to draw a composition node.
            - comp: current Composition
            - x, y: top-left coordinates
            - w, h: width and height to draw this node
            """
            # Add a tiny gap for stacked compositions

            if comp.is_leaf():
                comp.leaf_char.draw(dwg, group, w, h, x, y, stroke, char_is_empty)

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
        draw_node(self, 0, 0, size, size)

        svg_obj = SvgObject(group, size, size, 0, 0)
        return svg_obj


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
    
    def sentence_to_svg_obj(self, dwg, sentence, size=200, stroke=5):
        words = sentence.split(' ')

        word_trees = []
        for w in words:
            t = self.construct_word_tree(w)
            word_trees.append(t)

        char_gap = size / 8
        dims = ((size * len(words)) + ((len(words) - 1) * char_gap), size)

        sentence_group = dwg.g()
        x = 0
        for i in range(len(word_trees)):
            t = word_trees[i]
            svg_obj = t.create_svg_obj(dwg, size, stroke)
            svg_obj.set_xy(x, 0)
            svg_obj.draw_to_group(sentence_group)
            x += size + char_gap

        return SvgObject(sentence_group, dims[0], dims[1])




class GoetianRuneDrawer:
    
    RADICALS_DEFS = {
        'a': Character(
                Glyph(
                    'M 0 0 L -7 -10 L 7 -10 Z'
                ),
                simplified_glyphs=[
                ]
            ),
        'b': Character(
                Glyph(
                    'M 0 -19 V 0 M -2 -19 L 0 -19 C 16 -19 15 -20 15 -13 M 2 0 L -2 0',
                    is_hollow=True,
                    padding=[0.05,0.1,0.05,0.15],
                ),
                simplified_glyphs=[
                    Glyph(
                        'M 0 -19 V 0 M -2 -19 L 0 -19 C 15 -19 15 0 0 0 L -2 0'
                    )
                ]
            ),
        'c': Character(
                Glyph(
                    'M 0 -22 L 0 -16 M 0 -10 L 0 -4 M 0 -19 C 14 -31 14 5 0 -7',
                    is_hollow=True,
                    padding=[0.15,0.2,0.15,0.1],
                )
            ),
        'd': Character(
                Glyph(
                    'M -22 0 L -16 0 M -10 0 L -4 0 M -19 0 C -31 -14 5 -14 -7 0',
                    is_hollow=True,
                    padding=[0.15,0.2,0.1,0.2],
                ),
            ),
        'e': Character(
                Glyph(
                    'M -18 -20 L 3 -20 L -7 -2 Z M -13 -8 L -2 -8',
                ),
            ),
        'f': Character(
                Glyph(
                    'M -17 -30 L -17 -9 M -20 -30 L -17 -30 C -8 -30 -8 -19 -17 -19',
                ),
            ),
        'g': Character(
                Glyph(
                    'M 0 22 L 0 16 M 0 10 L 0 4 M 0 19 C -14 31 -14 -5 0 7',
                    is_hollow=True,
                    padding=[0.15,0.15,0.15,0.2],
                ),
            ),
        'h': Character(
                Glyph(
                    'M 18 20 L -3 20 L 7 2 Z M 2 18 L 2 4',
                ),
            ),
        'i': Character(
                Glyph(
                    'M 0 0 C -11 0 -11 15 0 15 C 11 15 11 0 0 0',
                ),
            ),
        'j': Character(
                Glyph(
                    'M -8 7 C -8 8 -8 15 0 15 C 11 15 9 -7 -6 2 M -8 7 L 4 7',
                ),
            ),
        'k': Character(
                Glyph(
                    'M -9 0 L -6 0 M -2 0 L 1 0 M -8 0 L -4 -7 L 0 0',
                ),
            ),
        'l': Character(
                Glyph(
                    'M -6 -9 L -6 -6 M -6 -8 L 4 -8 L 4 5',
                    is_hollow=True,
                    padding=[0.1,0.1,0,0.05],
                ),
            ),
        'm': Character(
                Glyph(
                    'M 6 8 L 6 6 M 6 8 L -4 8 L -4 -5',
                    is_hollow=True,
                    padding=[0,0,0.1,0.1],
                ),
            ),
        'n': Character(
                Glyph(
                    'M -8 -8 L -8 8 L 0 -8 L 0 8',
                ),
            ),
        'o': Character(
                Glyph(
                    'M 0 0 C 8 0 8 12 0 12 C -8 12 -8 0 0 0 M 0 0 L 0 12',
                ),
            ),
        'p': Character(
                Glyph(
                    'M 0 -14 L 0 6 M -2 6 L 0 6 C 8 6 8 -6 0 -6 M -2 -14 L 3 -14',
                ),
            ),
        'q': Character(
                Glyph(
                    'M 0 -5 L 5 0 L 0 5 L -5 0 Z',
                ),
            ),
        'r': Character(
                Glyph(
                    'M -8 0 L -8 -20 L 8 -20 L 8 0 M -9 0 L -6 0 M 9 0 L 6 0',
                    is_hollow=True,
                    padding=[0.05,0.1,0.05,0.1],
                ),
            ),
        's': Character(
                Glyph(
                    'M 0 0 C 8 0 8 12 0 12 C -8 12 -8 0 0 0 M -6 6 L 6 6',
                ),
            ),
        't': Character(
                Glyph(
                    'M -5 4 L -1 8 L 3 4 M -1 8 L -1 14',
                ),
            ),
        'u': Character(
                Glyph(
                    'M 0 2 L -3 5 L 0 8',
                ),
            ),
        'v': Character(
                Glyph(
                    'M 9 -0 L 6 -0 M 2 -0 L -1 0 M 8 -0 L 4 7 L 0 0',
                ),
            ),
        'w': Character(
                Glyph(
                    'M -9 -4 L -1 -4 L -9 4',
                ),
            ),
        'x': Character(
                Glyph(
                    'M -9 -4 L -1 -4 L -1 4 L -9 4 Z',
                    is_hollow=False,
                    padding=[0.1,0.1,0.1,0.1],
                ),
            ),
        'y': Character(
                Glyph(
                    'M -3 -7 L -12 2 M -12 -7 L -5 0',
                ),
            ),
        'z': Character(
                Glyph(
                    'M -13 -3 L -9 -3 L -13 -7 L -1 -7 L -5 -3 L -1 -3',
                ),
            ),
    }
    
    def __init__(self):
        pass

    def chunk_cv(self, word, allowed_chars=None):
        """
        Break word into contiguous consonant-vowel chunks.
        Leading vowels are lumped with the first consonant cluster.
        Skips characters not in allowed_chars.
        """
        if allowed_chars is None:
            allowed_chars = set("abcdefghijklmnopqrstuvwxyz")
        
        vowels = set("aeiouy")
        chunks = []
        i = 0
        n = len(word)
        
        current_chunk = ""
        first_chunk = True

        while i < n:
            c = word[i].lower()
            i += 1

            if c not in allowed_chars:
                # skip unallowed characters without breaking the chunk
                continue

            if not current_chunk:
                # Start a new chunk
                current_chunk += c
            else:
                # Determine last char type
                last_char = current_chunk[-1].lower()
                last_is_vowel = last_char in vowels
                current_is_vowel = c in vowels

                if last_is_vowel and not current_is_vowel:
                    # Start a new chunk when switching from vowel->consonant
                    chunks.append(current_chunk)
                    current_chunk = c
                else:
                    current_chunk += c

        # Append the last chunk
        if current_chunk:
            chunks.append(current_chunk)

        # Special handling for leading vowels at the start
        if first_chunk and chunks:
            if chunks[0][0].lower() in vowels and len(chunks) > 1:
                # merge leading vowels with next consonant cluster
                chunks[0] += chunks.pop(1)

        return chunks



    def construct_word_tree(self, word):

        # first get the components
        components = list(word)
        trees = []
        # for each component check first if it is a radical
        for component in components:
            if component in self.RADICALS_DEFS:
                trees.append(Composition(leaf_char=self.RADICALS_DEFS[component]))
            else:
                print(f"WARNING: '{component}' not a radical or word")
        
        l = len(trees)
        
        if l == 0:
            return None
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


    def sentence_to_svg_obj(self, dwg, sentence, size=200, stroke=5):
        
        # first split the sentance up into chunks words
        words = sentance.split(' ')

        # now add all the chunks together
        chunks = []
        for word in words:
            word_chunks = self.chunk_cv(word.lower())
            chunks += word_chunks

        chunk_trees = []
        for chunk in chunks:
            chunk_tree = self.construct_word_tree(chunk)
            if chunk_tree is None:
                continue
            chunk_trees.append(chunk_tree)

        char_gap = size / 8
        dims = ((size * len(chunk_trees)) + ((len(chunk_trees) - 1) * char_gap), size)

        sentence_group = dwg.g()
        x = 0
        for i in range(len(chunk_trees)):
            t = chunk_trees[i]
            svg_obj = t.create_svg_obj(dwg, size, stroke)
            svg_obj.set_xy(x, 0)
            svg_obj.draw_to_group(sentence_group)
            x += size + char_gap

        return SvgObject(sentence_group, dims[0], dims[1])


class GoetianSigilRingDrawer:

    def __init__(self):
        self.rune_drawer = GoetianRuneDrawer()

    def create_polygon_with_circle(self, n_sides, radius, stroke="black", fill_polygon="none", fill_circle="none", stroke_width=1):
        """
        Create an SvgObject containing a regular N-sided polygon and an inscribed circle of radius R.
        """
        if n_sides < 3:
            raise ValueError("Polygon must have at least 3 sides")
        
        dwg = svgwrite.Drawing()
        group = dwg.g()  # Group to hold polygon + circle

        # Center at (radius, radius)
        cx, cy = radius, radius

        # --- Polygon ---
        points = []
        for i in range(n_sides):
            angle = 2 * math.pi * i / n_sides - math.pi/2  # start pointing up
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            points.append((x, y))
        
        poly = dwg.polygon(points=points, stroke=stroke, fill=fill_polygon, stroke_width=stroke_width)
        group.add(poly)

        # --- Inscribed circle ---
        # circ = dwg.circle(center=(cx, cy), r=radius, stroke=stroke, fill=fill_circle, stroke_width=stroke_width)
        # group.add(circ)

        # SvgObject width/height = 2*radius to fully contain the shape
        obj = SvgObject(group=group, width=2*radius, height=2*radius, center_x=radius, center_y=radius)
        return obj

    def sentence_to_svg_obj(self, dwg, sentence, radius=1000, run_size=200, stroke=5):

        words = sentance.split(' ')

        # save some room for the runes


        return self.create_polygon_with_circle(len(words), size/2, stroke_width=stroke)

class GoetianSigilCenterDrawer:
    def __init__(self):
        self.rune_drawer = GoetianRuneDrawer()
    
    def create_star_outer_triangles(self, n_triangles, base_length, stroke="black", fill="none", stroke_width=1):
        """
        Create an SvgObject of a star made of n outer triangles with given base length.
        
        :param n_triangles: Number of triangles around the circle
        :param base_length: Length of each triangle base
        """
        if n_triangles < 2:
            raise ValueError("Need at least 2 triangles")
        
        dwg = svgwrite.Drawing()
        group = dwg.g()
        
        # Compute the radius of the circle where triangle bases lie
        # Formula: radius = base_length / (2 * sin(pi / n_triangles))
        r_base = base_length / (2 * math.sin(math.pi / n_triangles))
        
        # Triangle height: choose same as base length for nice proportions
        h = base_length  # you can tweak this
        
        points_all = []

        for i in range(n_triangles):
            # Center angle for this triangle
            theta = 2 * math.pi * i / n_triangles

            # Base center coordinates on the circle
            bx = r_base * math.cos(theta)
            by = r_base * math.sin(theta)

            # Base endpoints (perpendicular to radius)
            perp_angle = theta + math.pi / 2
            half_base = base_length / 2
            x1 = bx + half_base * math.cos(perp_angle)
            y1 = by + half_base * math.sin(perp_angle)
            x2 = bx - half_base * math.cos(perp_angle)
            y2 = by - half_base * math.sin(perp_angle)

            # Tip coordinates (along radius)
            tip_length = h
            tx = bx + tip_length * math.cos(theta)
            ty = by + tip_length * math.sin(theta)

            tri = dwg.polygon(points=[(x1, y1), (x2, y2), (tx, ty)],
                            stroke=stroke, fill=fill, stroke_width=stroke_width)
            group.add(tri)
            points_all.extend([(x1, y1), (x2, y2), (tx, ty)])

        # Compute bounding box
        xs = [p[0] for p in points_all]
        ys = [p[1] for p in points_all]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x
        height = max_y - min_y
        center_x = width / 2 - min_x
        center_y = height / 2 - min_y

        # Shift group so top-left at (0,0)
        group.translate(-min_x, -min_y)

        obj = SvgObject(group=group, width=width, height=height, center_x=center_x, center_y=center_y)
        return obj
    def sentence_to_svg_obj(self, dwg, sentence, radius=1000, run_size=200, stroke=5):
        return self.create_star_outer_triangles(8, 100, stroke_width=stroke)

sentance = 'the quick brown fox jumped over the lazy dog'

drawer = GoetianSigilCenterDrawer()
# drawer = LogogramDrawer()

s = 400
dwg = svgwrite.Drawing('out.svg')
dwg.add(dwg.rect(insert=(0, 0), size=("100%", "100%"), fill="white"))
s = drawer.sentence_to_svg_obj(dwg, sentance)
dwg['width'] = s.width
dwg['height'] = s.height
s.draw_to_canvas(dwg)
dwg.save()
