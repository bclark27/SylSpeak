from svgpathtools import parse_path
from svgpathtools import Path
from enum import Enum
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
                name, 
                path_str, 
                width, 
                height,
                min_width_ratio=0.5,
                max_width_ratio=1.5,
                min_height_ratio=0.5,
                max_height_ratio=1.5,
                is_hollow=False,
                inside_margins=[0.1, 0.1, 0.1, 0.1]):
        self.name = name
        self.path = parse_path(path_str)
        self.width = width
        self.height = height
        self.min_width_ratio = min_width_ratio
        self.max_width_ratio = max_width_ratio
        self.min_height_ratio = min_height_ratio
        self.max_height_ratio = max_height_ratio
        self.is_hollow = is_hollow
        self.inside_margins = inside_margins

class Composition:
    def __init__(self, op=None, sub_comp1=None, sub_comp2=None, leaf_glyph=None):
        self.op = op
        self.sub_comp1 = sub_comp1
        self.sub_comp2 = sub_comp2
        self.leaf_glyph = leaf_glyph
        self.sub_comp1_percent = 0.5

    def is_leaf(self):
        return self.leaf_glyph is not None

    def get_composition_side_ratio(self):
        if self.is_leaf():
            return (self.leaf_glyph.width, self.leaf_glyph.height)
        return (1, 1)

    def get_stretching_limits(self):
        if self.is_leaf():
            g = self.leaf_glyph
            return (g.min_width_ratio, g.max_width_ratio,
                    g.min_height_ratio, g.max_height_ratio)

        w1_min, w1_max, h1_min, h1_max = self.sub_comp1.get_stretching_limits()
        w2_min, w2_max, h2_min, h2_max = self.sub_comp2.get_stretching_limits()

        if self.op == CompositionOp.HORZ:
            min_width = w1_min + w2_min
            max_width = w1_max + w2_max
            min_height = max(h1_min, h2_min)
            max_height = min(h1_max, h2_max)
        elif self.op == CompositionOp.VERT:
            min_height = h1_min + h2_min
            max_height = h1_max + h2_max
            min_width = max(w1_min, w2_min)
            max_width = min(w1_max, w2_max)
        elif self.op == CompositionOp.IN:
            min_width = min(w1_min, w2_min)
            max_width = max(w1_max, w2_max)
            min_height = min(h1_min, h2_min)
            max_height = max(h1_max, h2_max)
        else:
            raise ValueError("Composition operation not set")

        return (min_width, max_width, min_height, max_height)

    def can_fit(self, target_w_ratio, target_h_ratio):
        """
        Returns True if this composition can accommodate the given width/height ratios
        (relative to its natural dimensions) without violating any child limits.
        """
        if self.is_leaf():
            g = self.leaf_glyph
            return (g.min_width_ratio <= target_w_ratio <= g.max_width_ratio and
                    g.min_height_ratio <= target_h_ratio <= g.max_height_ratio)

        # Recursive: get limits of sub-compositions
        w1_min, w1_max, h1_min, h1_max = self.sub_comp1.get_stretching_limits()
        w2_min, w2_max, h2_min, h2_max = self.sub_comp2.get_stretching_limits()

        # Check composition constraints
        if self.op == CompositionOp.HORZ:
            # Target width is shared across the two children
            # (simplified: just check if total fits)
            can_w = (w1_min + w2_min <= target_w_ratio <= w1_max + w2_max)
            can_h = (max(h1_min, h2_min) <= target_h_ratio <= min(h1_max, h2_max))

        elif self.op == CompositionOp.VERT:
            can_h = (h1_min + h2_min <= target_h_ratio <= h1_max + h2_max)
            can_w = (max(w1_min, w2_min) <= target_w_ratio <= min(w1_max, w2_max))

        elif self.op == CompositionOp.IN:
            can_w = (min(w1_min, w2_min) <= target_w_ratio <= max(w1_max, w2_max))
            can_h = (min(h1_min, h2_min) <= target_h_ratio <= max(h1_max, h2_max))
        else:
            raise ValueError("Composition operation not set")

        return can_w and can_h
        """
        Recursively determine the allowable width and height ratio ranges
        for this composition node.
        Returns (min_width_ratio, max_width_ratio, min_height_ratio, max_height_ratio)
        """
        if self.is_leaf():
            g = self.leaf_glyph
            return (g.min_width_ratio, g.max_width_ratio,
                    g.min_height_ratio, g.max_height_ratio)

        # Recursively get child limits
        w1_min, w1_max, h1_min, h1_max = self.sub_comp1.get_stretching_limits()
        w2_min, w2_max, h2_min, h2_max = self.sub_comp2.get_stretching_limits()

        if self.op == CompositionOp.HORZ:
            # Horizontal stack: widths add, heights must match
            min_width = w1_min + w2_min
            max_width = w1_max + w2_max
            min_height = max(h1_min, h2_min)
            max_height = min(h1_max, h2_max)

        elif self.op == CompositionOp.VERT:
            # Vertical stack: heights add, widths must match
            min_height = h1_min + h2_min
            max_height = h1_max + h2_max
            min_width = max(w1_min, w2_min)
            max_width = min(w1_max, w2_max)

        elif self.op == CompositionOp.IN:
            # Nested composition: constrained by outer glyph
            # Use inner’s flexibility limited by outer’s interior space
            min_width = min(w1_min, w2_min)
            max_width = max(w1_max, w2_max)
            min_height = min(h1_min, h2_min)
            max_height = max(h1_max, h2_max)

        else:
            raise ValueError("Composition operation not set")

        return (min_width, max_width, min_height, max_height)

    def is_hollow(self):
        return self.is_leaf() and self.leaf_glyph.is_hollow

    def calc_constructions(self):
        if self.is_leaf():
            self.op = CompositionOp.NONE
            return
        
        self.sub_comp1.calc_constructions()
        self.sub_comp2.calc_constructions()

        # if the outer comp is hollow
        # and the comp that would be inside is not also doing a nesting in
        # then we can do nesting here
        if self.sub_comp1.is_hollow() and self.sub_comp2.op != CompositionOp.IN:
            self.op = CompositionOp.IN
            return

        # now we need to check if the horozontal or the vertical will cause less stretching problems
        stretch1 = self.sub_comp1.get_stretching_limits()
        stretch2 = self.sub_comp2.get_stretching_limits()

        # TODO: need to calculate here which vertical or horizontal to do
        # also calculate the perportions of the square that will be assigned to each half


GLYPHS = {
    'w': Glyph('w', 
        'M 0 0 L 0 10 M 0 0 L 10 0 M 10 0 L 10 10',
        10,
        5,
        is_hollow=True),
    's': Glyph('s', 
        'M 0 0 L 0 10 M 0 0 L 10 0 M 10 0 L 10 10 M 0 10 L 10 10',
        10,
        10,
        is_hollow=True)
}

comp = Composition(
    sub_comp1=Composition(leaf_glyph=GLYPHS['w']),
    sub_comp2=Composition(leaf_glyph=GLYPHS['s'])
)

comp.calc_constructions()

print(comp.sub_comp1.get_ratio_and_flex())