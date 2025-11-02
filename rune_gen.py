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
        self.origin_rel_x = 0
        self.origin_rel_y = 0

        self._update_transform()

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

    use_detailed_glyphs = True
    detailed_steps = 30

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
        """Draws each segment of the glyph path, thickening from min to max stroke width within each segment."""

        p = self.path

        # --- Scale & position path as before ---
        minx, maxx, miny, maxy = p.bbox()
        path_width = maxx - minx
        path_height = maxy - miny

        scale_x = draw_w / path_width
        scale_y = draw_h / path_height
        p = p.scaled(scale_x, scale_y)
        p = p.scaled(1, -1)
        p = p.scaled(1 - (self.margin[1] + self.margin[3]), 1 - (self.margin[0] + self.margin[2]))
        p = p.translated(complex(x, y + draw_h))
        p = p.translated(complex(draw_w * self.margin[3], -draw_h * self.margin[2]))
        p = p.translated(complex(draw_w * self.translate[0], draw_h * self.translate[1]))

        if not self.use_detailed_glyphs:
            parent.add(dwg.path(
                d=p.d(), 
                stroke='black', 
                fill='none', 
                stroke_width=stroke
            ))

            return

        g = dwg.g()

        min_stroke = 2
        max_stroke = stroke

        # --- For each SVG segment ---
        for seg in p:
            prev_point = seg.point(0)

            for i in range(1, self.detailed_steps + 1):
                t0 = (i - 1) / self.detailed_steps
                t1 = i / self.detailed_steps

                # Sample along the segment
                p0 = seg.point(t0)
                p1 = seg.point(t1)

                # Local interpolation of stroke width
                local_t = t1
                stroke_w = min_stroke + (max_stroke - min_stroke) * local_t

                # Draw this little line piece
                g.add(dwg.line(
                    start=(p0.real, p0.imag),
                    end=(p1.real, p1.imag),
                    stroke='black',
                    stroke_width=stroke_w,
                    stroke_linecap='round'
                ))
            
            if seg == p[-1]:  # only for the last path segment
                end_point = seg.point(1)
                tangent = seg.derivative(1)
                angle = math.atan2(tangent.imag, tangent.real)

                # --- Configure flick behavior ---
                flick_angle = angle - math.radians(random.uniform(20, 40))  # angled back a bit
                flick_len = 0.2 * draw_w                              # small compared to total width
                steps = 10                                                   # steps along flick line

                # Compute flick path points
                for i in range(steps):
                    t0 = i / steps
                    t1 = (i + 1) / steps

                    start_p = end_point + cmath.rect(flick_len * t0, flick_angle)
                    end_p   = end_point + cmath.rect(flick_len * t1, flick_angle)

                    # taper stroke width from max_stroke → 0
                    stroke_w = max_stroke * (1 - t1)

                    parent.add(dwg.line(
                        start=(start_p.real, start_p.imag),
                        end=(end_p.real, end_p.imag),
                        stroke='black',
                        stroke_width=stroke_w,
                        stroke_linecap='round',
                        opacity=1 - t1  # fade out a bit
                    ))


        parent.add(g)

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

        svg_obj = SvgObject(group, size, size)
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

class GoetianRuneDrawer:
    
    RUNES1 = {
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
    
    SVG_CHARS = RUNES1

    def __init__(self):
        pass

    def chunk_cv(self, word, allowed_chars=None):
        # n = 2
        # return [word[i:i + n] for i in range(0, len(word), n)]
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
            if component in self.SVG_CHARS:
                trees.append(Composition(leaf_char=self.SVG_CHARS[component]))
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

    def draw_to_svg_objs(self, sentence, size=200, stroke_width=5):
        
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

        dwg = svgwrite.Drawing()
        x = 0
        svgs = []
        for i in range(len(chunk_trees)):
            t = chunk_trees[i]
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

def draw_ring(group, radius, x, y, stroke="black", stroke_width=5, fill="none"):
    dwg = svgwrite.Drawing()
    circle = dwg.circle(center=(x, y), r=radius, stroke=stroke, fill=fill, stroke_width=stroke_width)
    group.add(circle)

def draw_line(group, x1, y1, x2, y2, stroke="black", stroke_width=5):
    """
    Create an SvgObject containing a line from (x1, y1) to (x2, y2).
    """
    dwg = svgwrite.Drawing()
    line = dwg.line(start=(x1, y1), end=(x2, y2), stroke=stroke, stroke_width=stroke_width)
    group.add(line)

    return group

def draw_poly_triangle(group, base, n, stroke_width=5):
    angle = 360 / n
    angle = (angle / 360) * 2 * math.pi
    h = base * math.tan(angle) / 2

    draw_line(group, 0, h, base, h, stroke_width=stroke_width)
    draw_line(group, base/2, 0, base, h, stroke_width=stroke_width)
    draw_line(group, base/2, 0, 0, h, stroke_width=stroke_width)

    return h

def point_on_circle(radius, center_x, center_y, num_points, index):
    """
    Returns the (x, y) coordinates of a point evenly spaced around a circle.

    :param radius: radius of the circle
    :param center_x: x-coordinate of circle center
    :param center_y: y-coordinate of circle center
    :param num_points: total number of points evenly spaced
    :param index: which point to return (0-based)
    :return: (x, y) coordinates of the point
    """
    if num_points < 1:
        raise ValueError("num_points must be at least 1")
    angle = 2 * math.pi * index / num_points - math.pi/2  # start from top
    x = center_x + radius * math.cos(angle)
    y = center_y + radius * math.sin(angle)
    return x, y

def distance_between_points(radius, num_points):
    if num_points < 2:
        raise ValueError("Need at least 2 points")
    return 2 * radius * math.sin(math.pi / num_points)

def ngon_vertices_from_incircle(R: float, N: int, center: Tuple[float,float]=(0.0,0.0), start_angle: float = -math.pi/2) -> Tuple[float, List[Tuple[float,float]], float]:
    """
    Given an incircle radius R and number of sides N, compute:
      - circumradius (distance from center to polygon vertices)
      - list of N vertex (x,y) coordinates (regular N-gon)
      - side length

    Parameters:
      R           : radius of the circle that the N-gon's sides are tangent to (inradius)
      N           : number of polygon sides (N >= 3)
      center      : (cx, cy) coordinates of the shared center
      start_angle : angular offset for vertex 0 in radians (default -pi/2 -> top)

    Returns:
      (circumradius, vertices_list, side_length)
    """
    if N < 3:
        raise ValueError("N must be >= 3")

    cx, cy = center
    # circumradius (distance to vertices)
    R_v = R / math.cos(math.pi / N)

    # side length
    side = 2 * R * math.tan(math.pi / N)  # equivalent to 2 * R_v * sin(pi/N)

    vertices = []
    for k in range(N):
        angle = start_angle + 2 * math.pi * k / N
        x = cx + R_v * math.cos(angle)
        y = cy + R_v * math.sin(angle)
        vertices.append((x, y))

    return R_v, vertices, side

def chunk_by_pattern(arr, pattern):
    result = []
    i = 0
    p_i = 0

    while i < len(arr):
        # Get current chunk size from pattern
        size = pattern[p_i]
        result.append(arr[i:i + size])
        i += size
        # Advance pattern index, looping if needed
        if p_i < len(pattern) - 1:
            p_i += 1
        # Once we reach the last pattern size, stay there
        # so it repeats forever
    return result

def star_with_construction(N: int, R: float, stroke=1, inner_r=-1) -> SvgObject:
    """
    Generates an SvgObject of an N-pointed star with optional construction lines.
    
    Parameters:
      N           : number of star points (>= 2)
      R           : outer radius
      center      : (cx, cy) center
      stroke      : stroke width
      inner_ratio : fraction of R for inner vertices (0 < inner_ratio < 1)
    """
    if N < 2:
        raise ValueError("N must be >= 2")
    cx, cy = (R, R)

    if inner_r >= R or inner_r <= 0:
        inner_r = R * 0.5
    
    points = []
    for i in range(2*N):
        angle = math.pi/2 + i * math.pi / N  # start at top
        radius = R if i % 2 == 0 else inner_r
        x = cx + radius * math.cos(angle)
        y = cy - radius * math.sin(angle)  # SVG y-axis down
        points.append((x,y))
    
    dwg = svgwrite.Drawing()
    g = dwg.g()
    
    # --- main star path ---
    path_data = "M {} {}".format(*points[0])
    for p in points[1:]:
        path_data += " L {} {}".format(*p)
    path_data += " Z"
    g.add(dwg.path(d=path_data, stroke="black", fill="none", stroke_width=stroke))
    
    # --- construction lines ---
    # connect inner vertices to form inner polygon
    inner_vertices = points[1::2]
    for i in range(len(inner_vertices)):
        x1, y1 = inner_vertices[i]
        x2, y2 = inner_vertices[(i+1) % len(inner_vertices)]
        g.add(dwg.line(start=(x1, y1), end=(x2, y2), stroke='black', stroke_width=stroke))
    
    # connect outer vertices to form outer polygon
    outer_vertices = points[0::2]
    for i in range(len(outer_vertices)):
        x1, y1 = outer_vertices[i]
        x2, y2 = outer_vertices[(i+1) % len(outer_vertices)]
        g.add(dwg.line(start=(x1, y1), end=(x2, y2), stroke='gray', stroke_width=stroke))
    
    width = height = 2*R
    return SvgObject(g, width, height)

class GoetianSigilRingDrawer:

    def __init__(self):
        pass

    def rune_group_to_ring(self, group, rune_svgs, center_x, center_y, inner_radius, rune_size, stroke_width, even_odd):

        # find the total width
        vert_rune_gap = rune_size / 8
        outer_radius = inner_radius + vert_rune_gap + rune_size + vert_rune_gap

        draw_ring(group, inner_radius, center_x, center_y, stroke_width=stroke_width)
        draw_ring(group, outer_radius, center_x, center_y, stroke_width=stroke_width)


        points = len(rune_svgs) * 2
        rune_idx = 0
        angle_spacing = 360 / points
        for i in range(points):
            if (i % 2 == 0) == even_odd:
                rune_point = point_on_circle(inner_radius + vert_rune_gap, center_x, center_y, points, i)
                rune_svg_obj = rune_svgs[rune_idx]
                rune_svg_obj.set_origin(0.5, 1)
                rune_svg_obj.set_xy(rune_point[0] - rune_size/2, rune_point[1] - rune_size)
                rune_svg_obj.set_rotate(angle_spacing * i)
                rune_svg_obj.draw_to_group(group)
                rune_idx += 1
            else:
                spoke_start = point_on_circle(inner_radius, center_x, center_y, points, i)
                spoke_end = point_on_circle(outer_radius, center_x, center_y, points, i)
                draw_line(group, 
                            spoke_start[0], spoke_start[1],
                            spoke_end[0], spoke_end[1],
                            stroke_width=stroke_width)
      
    def draw_to_svg_obj(self, rune_svgs, inner_radius=200, rune_size=200, stroke_width=5):

        # get all the runes we will place
        max_rune_per_ring = 12
        rune_svg_groups = [rune_svgs[i:i + max_rune_per_ring] for i in range(0, len(rune_svgs), max_rune_per_ring)]
        
        dwg = svgwrite.Drawing()
        group = dwg.g()

        rune_group_count = len(rune_svg_groups)
        vert_rune_gap = rune_size / 8
        ring_thickness = vert_rune_gap + rune_size + vert_rune_gap

        full_radius = rune_group_count * ring_thickness + inner_radius
        center_x = full_radius
        center_y = full_radius

        for i in range(rune_group_count):
            this_ring_inner_r = (i * ring_thickness) + inner_radius
            self.rune_group_to_ring(group, rune_svg_groups[i], center_x, center_y, this_ring_inner_r, rune_size, stroke_width, i % 2 == 0)

        return SvgObject(group, full_radius*2, full_radius*2)

class GoerianSigilPolyDrawer:
    def __init__(self):
        pass

    def draw_to_svg_obj(self, rune_svgs, inner_radius=800, rune_size=200, stroke_width=5):
        dwg = svgwrite.Drawing()
        group = dwg.g()
        # b = rune_size * 2.5
        # h = draw_poly_triangle(group, b, len(rune_svgs), stroke_width=stroke_width)
        # r = rune_svgs[0]
        # r.set_xy(b/2 - rune_size/2,b - rune_size/2)
        # r.draw_to_group(group)

        rune_count = len(rune_svgs)

        Rv, verts, s = ngon_vertices_from_incircle(inner_radius, rune_count)
        
        # ok now we know the max corner dist as well as the rune size and placement so we can get teh theoretical full width of this thing
        rune_gap = rune_size / 8
        rune_dist = inner_radius + rune_gap
        max_rune_dist = rune_dist + rune_size
        outer_poly_dist = Rv
        full_width = max(max_rune_dist, outer_poly_dist) * 2

        center_x = full_width / 2
        center_y = full_width / 2

        # ok now we want to draw the polygon part and the runes
        # so we want to alternate the points vertex,rune,vertex,rune
        # so if we put the runes on the even, then the vertexes should go on the odd

        # draw the poly first
        for i in range(rune_count):
            vert1_x, vert1_y = point_on_circle(outer_poly_dist, center_x, center_y, rune_count, i)
            vert2_x, vert2_y = point_on_circle(outer_poly_dist, center_x, center_y, rune_count, i+1)
            draw_line(group, vert1_x, vert1_y, vert2_x, vert2_y, stroke_width=stroke_width)

        rune_angle_spacing = 360 / rune_count
        for i in range(rune_count):
            x, y = point_on_circle(rune_dist, center_x, center_y, rune_count * 2, i * 2 + 1)
            rune_svgs[i].set_rotate((rune_angle_spacing * i) + (rune_angle_spacing / 2))
            rune_svgs[i].set_origin(0.5, 1)
            rune_svgs[i].set_xy(x - rune_size/2, y - rune_size)
            rune_svgs[i].draw_to_group(group)
            
        return SvgObject(group, full_width, full_width)

class GoerianSigilDrawer:
    def __init__(self):
        self.ring_drawer = GoetianSigilRingDrawer()
        self.poly_drawer = GoerianSigilPolyDrawer()
        self.rune_drawer = GoetianRuneDrawer()

    def draw_to_svg_obj(self, sentence, rune_size=200, stroke_width=10, pattern=[6,3,7,4,6,4,7,13,17,6]):

        rune_svgs = self.rune_drawer.draw_to_svg_objs(sentence, rune_size, stroke_width)

        dwg = svgwrite.Drawing()
        group = dwg.g()
\
        rune_svg_groups = chunk_by_pattern(rune_svgs, pattern)
       
        sigil_svgs = []
        start_radius = rune_size * 4
        current_max_radius = start_radius
        ring_space = rune_size / 8
        
        group_count = len(rune_svg_groups)
        for i in range(group_count):
            rune_group = rune_svg_groups[i]
            sigil_svg = None

            if i == 0:
                center_peice_svg = star_with_construction(len(rune_group), start_radius, stroke=stroke_width)
                sigil_svgs.append(center_peice_svg)

            if i != 0 and i % 6 == 0:
                start_deco = star_with_construction(len(rune_group), current_max_radius * 1.2, inner_r=current_max_radius, stroke=stroke_width)
                sigil_svgs.append(start_deco)
                current_max_radius = start_deco.width / 2

            if i % 3 == 0:
                sigil_svg = self.poly_drawer.draw_to_svg_obj(rune_group, inner_radius=current_max_radius, stroke_width=stroke_width, rune_size=rune_size)
                current_max_radius = sigil_svg.width / 2 + ring_space
            elif i % 3 == 1:
                sigil_svg = self.ring_drawer.draw_to_svg_obj(rune_group, inner_radius=current_max_radius, stroke_width=stroke_width, rune_size=rune_size)
                current_max_radius = sigil_svg.width / 2 + ring_space
            elif i % 3 == 2:
                current_max_radius += ring_space
                sigil_svg = self.ring_drawer.draw_to_svg_obj(rune_group, inner_radius=current_max_radius, stroke_width=stroke_width, rune_size=rune_size)
                current_max_radius = sigil_svg.width / 2 + ring_space

            sigil_svgs.append(sigil_svg)

        center_x = current_max_radius
        center_y = current_max_radius

        for sigil_svg in sigil_svgs:
            sigil_svg.set_xy(center_x - sigil_svg.width/2, center_y - sigil_svg.height/2)
            sigil_svg.draw_to_group(group)

        return SvgObject(group, current_max_radius*2, current_max_radius*2)

        # poly1 = self.poly_drawer.draw_to_svg_obj(rune_svgs, stroke_width=stroke_width, rune_size=rune_size)
        # poly1.draw_to_group(group)
        # return SvgObject(group, poly1.width, poly1.height)

        # here move the rune generation and count out here
        # group the syllables for each concentric ring
        # intersparce with some astrology runes
        # add a center peice





sentance = "We live on a placid island of ignorance in the midst of black seas of infinity, and it was not meant that we should voyage far."

drawer = GoerianSigilDrawer()
#drawer = LogogramDrawer()

s = 400
dwg = svgwrite.Drawing('out.svg')
dwg.add(dwg.rect(insert=(0, 0), size=("100%", "100%"), fill="white"))
s = drawer.draw_to_svg_obj(sentance)
dwg['width'] = s.width
dwg['height'] = s.height
s.draw_to_canvas(dwg)
dwg.save()
