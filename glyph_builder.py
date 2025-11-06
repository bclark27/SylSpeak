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
