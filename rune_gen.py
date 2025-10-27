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
                name, 
                path_str, 
                is_hollow=False,
                inside_margins=[0.15, 0.15, 0.15, 0.15], # top right bot left
                tall_path_str=None,
                wide_path_str=None
                ):
        self.name = name
        self.is_hollow = is_hollow
        self.inside_margins = inside_margins

        self.path = parse_path(path_str)
        
        minx, maxx, miny, maxy = self.path.bbox()
        self.width = maxx - minx
        self.height = maxy - miny

        cx = self.width / 2
        cy = self.height / 2
        self.path = (
            self.path
            .translated(-complex(cx, cy))
            .scaled(-1, -1)
            .translated(complex(cx, cy))
        )

        self.path = self.path.scaled(-1, 1)

        minx, maxx, miny, maxy= self.path.bbox()
        self.path = self.path.translated(-complex(minx, miny))

        

class Composition:
    def __init__(self, op=None, sub_comp1=None, sub_comp2=None, leaf_glyph=None):
        self.op = op
        self.sub_comp1 = sub_comp1
        self.sub_comp2 = sub_comp2
        self.leaf_glyph = leaf_glyph
        self.sub_comp1_percent = 0.5

    def is_leaf(self):
        return self.leaf_glyph is not None

    def is_hollow(self):
        return self.is_leaf() and self.leaf_glyph.is_hollow

    def get_composition_side_ratio(self):
        if self.is_leaf():
            return (self.leaf_glyph.width, self.leaf_glyph.height)

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


    def calc_constructions(self):
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



    def draw_svg(self, filename, size=400, stroke=5, gap=0.02):
        """
        Draws this composition as an independent SVG.
        - size: final image size in pixels (square)
        - stroke: stroke width in pixels
        - gap: fraction of square size to leave as gap between stacked glyphs
        """
        dwg = svgwrite.Drawing(filename, size=(size, size))
        dwg.add(dwg.rect(insert=(0, 0), size=(size, size), fill='white'))

        def draw_node(comp, x, y, w, h):
            """
            Recursive helper to draw a composition node.
            - comp: current Composition
            - x, y: top-left coordinates
            - w, h: width and height to draw this node
            """
            # Add a tiny gap for stacked compositions
            gap_w = w * gap
            gap_h = h * gap
            draw_w = w - gap_w
            draw_h = h - gap_h
            offset_x = x + gap_w / 2
            offset_y = y + gap_h / 2

            if comp.is_leaf():
                path = comp.leaf_glyph.path
                # Scale path to fit draw_w x draw_h
                scale_x = draw_w / comp.leaf_glyph.width
                scale_y = draw_h / comp.leaf_glyph.height
                # Flip Y for SVG coordinates
                path = path.scaled(scale_x, -scale_y)
                # Translate to top-left of allocated box
                transformed_path = path.translated(complex(offset_x, offset_y + draw_h))
                dwg.add(dwg.path(d=transformed_path.d(), stroke='black', fill='none', stroke_width=stroke))


            elif comp.op == CompositionOp.VERT:
                # Vertical stack: divide height according to sub_comp1_percent
                h1 = draw_h * comp.sub_comp1_percent
                h2 = draw_h - h1
                draw_node(comp.sub_comp1, offset_x, offset_y, draw_w, h1)
                draw_node(comp.sub_comp2, offset_x, offset_y + h1, draw_w, h2)
            elif comp.op == CompositionOp.HORZ:
                # Horizontal stack: divide width according to sub_comp1_percent
                w1 = draw_w * comp.sub_comp1_percent
                w2 = draw_w - w1
                draw_node(comp.sub_comp1, offset_x, offset_y, w1, draw_h)
                draw_node(comp.sub_comp2, offset_x + w1, offset_y, w2, draw_h)
            elif comp.op == CompositionOp.IN:
                # IN operation: sub_comp2 is nested inside sub_comp1
                # Apply inside margins from the leaf glyph (assume sub_comp1 is hollow)
                outer = comp.sub_comp1
                inner = comp.sub_comp2
                if outer.is_leaf():
                    m_top, m_right, m_bottom, m_left = outer.leaf_glyph.inside_margins
                else:
                    m_top = m_right = m_bottom = m_left = 0.05  # default margin if not leaf
                # Convert margins to pixel space
                inner_x = offset_x + draw_w * m_left
                inner_y = offset_y + draw_h * m_top
                inner_w = draw_w * (1 - m_left - m_right)
                inner_h = draw_h * (1 - m_top - m_bottom)
                draw_node(outer, offset_x, offset_y, draw_w, draw_h)
                draw_node(inner, inner_x, inner_y, inner_w, inner_h)
            else:
                raise ValueError(f"Unknown composition operation: {comp.op}")

        # Start recursive drawing
        draw_node(self, 0, 0, size, size)
        dwg.save()


GLYPHS = {
    'w': Glyph('w', 
        'M 0 11 L 0 0 L 11 0 L 11 11 M 11 10 L 0 10',
        is_hollow=True,
        inside_margins=[0.05, 0.05, 0.15, 0.05]),
    's': Glyph('s', 
        'M 0 0 L 0 10 M 0 0 L 10 0 M 10 0 L 10 10 M 0 10 L 10 10',
        is_hollow=True),
    'g': Glyph('g', 
        'M 8 25 C 9 21 9 10.3333 9 3 M 4 7 C 5.6667 6.3333 7.3333 4.6667 9 3 C 10 1.6667 11 0.3333 11 -1',
        is_hollow=False),
    'm': Glyph('m', 
        'M 4 0 L 4 4 M 0 4 L 0 2 M 0 0 L 4 0',
        is_hollow=True),
    'p': Glyph('p', 
        'M 20 15 C 17 22 13 26 6 33 M 28 31 C 22 31 16 31 7 32 M 29 33 C 28 31 26 28 25 25 M 14 19 C 12 22 11 24 8 27',
        is_hollow=False),
    'f': Glyph('f', 
        'M 17 -15 C 16 -9 11 -4 6 -1 M 16 -12 L 23 -12 C 21 -4 16 3 4 7 M 18 -2 L 14 -5',
        is_hollow=False),
    'k': Glyph('k', 
        'M 6 13 L 6 -47 M 2 -53 C 6 -51 8 -49 9 -47 M 10 -51 L 59 -51 L 59 10 C 59 13 57 14 54 14 L 51 14',
        tall_path_str='M 12 -10 C 19 -22 24 -34 27 -48 M 12 -46 C 16 -35 20 -23 27 -13',
        is_hollow=True,
        inside_margins=[0.1, 0.1, 0.05, 0.15]),
    'h': Glyph('h', 
        'M 13 0 C 14 -3 16 -1 16 -28 L 40 -28 L 40 -23 L 16 -23 M 29 -28 C 29 -29 29 -29 28 -30',
        is_hollow=True,
        inside_margins=[0.27, 0.02, 0.02, 0.15])
}

def create_glyph_tree(word):
    random.seed(word)

    comps = []
    for i in range(len(word)):
        if word[i] not in GLYPHS:
            continue
        comps.append(Composition(leaf_glyph=GLYPHS[word[i]]))

    l = len(comps)
    while l > 1:
        idx = random.randint(0, l - 2)
        comp1 = comps[idx]
        comp2 = comps.pop(idx + 1)
        newComp = Composition(
            sub_comp1=comp1,
            sub_comp2=comp2
        )

        comps[idx] = newComp
        l = len(comps)

    return comps[0]

comp = create_glyph_tree('kk')
# comp = Composition(
#     sub_comp1=Composition(leaf_glyph=GLYPHS['w']),
#     sub_comp2=Composition(leaf_glyph=GLYPHS['g'])
# )

comp.calc_constructions()
comp.draw_svg('out.svg')