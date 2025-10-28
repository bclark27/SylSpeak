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
                padding=[0.15, 0.15, 0.15, 0.15], # top right bot left (interior space)
                margin=[0.05, 0.05, 0.05, 0.05], # top right bot left (outside)
                translate=[0, 0], # x y (translate based on fraction of the width of the final area)
                ):
        self.name = name
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

    '''
    def closest_aspect_ratio_index(paths, ref_hw):
        """
        Returns the index in hw_list whose aspect ratio (h/w)
        is closest to that of ref_hw (reference height/width tuple).
        """
        ref_h, ref_w = ref_hw
        ref_ratio = ref_h / ref_w

        closest_idx = None
        closest_diff = float('inf')

        for i, p in enumerate(paths):
            if p is None:
                continue
            minx, maxx, miny, maxy = p.bbox()
            w = maxx - minx
            h = maxy - miny
            ratio = h / w
            diff = abs(ratio - ref_ratio)
            if diff < closest_diff:
                closest_diff = diff
                closest_idx = i

        return closest_idx

    def get_best_path(self, ref_hw):
        options = [self.path] + self.simplified_paths
        return options[Glyph.closest_aspect_ratio_index(options, ref_hw)]

    '''
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

    def draw_svg(self, dwg, pos_x, pos_y, size=400, stroke=5):
        """
        Draws this composition as an independent SVG.
        - size: final image size in pixels (square)
        - stroke: stroke width in pixels
        """
        def draw_node(comp, x, y, w, h, glyph_is_filled=False):
            """
            Recursive helper to draw a composition node.
            - comp: current Composition
            - x, y: top-left coordinates
            - w, h: width and height to draw this node
            """
            # Add a tiny gap for stacked compositions

            if comp.is_leaf():
                comp.leaf_glyph.draw(dwg, w, h, x, y, stroke)

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
                    m_top, m_right, m_bottom, m_left = outer.leaf_glyph.padding
                else:
                    m_top = m_right = m_bottom = m_left = 0.05  # default margin if not leaf
                # Convert margins to pixel space
                inner_x = x + w * m_left
                inner_y = y + h * m_top
                inner_w = w * (1 - m_left - m_right)
                inner_h = h * (1 - m_top - m_bottom)
                draw_node(outer, x, y, w, h, True)
                draw_node(inner, inner_x, inner_y, inner_w, inner_h)
            else:
                raise ValueError(f"Unknown composition operation: {comp.op}")

        # Start recursive drawing
        draw_node(self, pos_x, pos_y, size, size)

GLYPHS = {
    's': Glyph('s', 
        'M 0 11 L 0 0 L 11 0 L 11 11 M 11 10 L 0 10',
        is_hollow=True,
        padding=[0.05, 0.05, 0.15, 0.05]),
    'w': Glyph('w', 
        'M 20 -34 C 23 -34 22 -47 22 -54 L 40 -54 L 40 -34',
        is_hollow=True,
        padding=[0.1, 0.1, 0.05, 0.2]),
    'g': Glyph('g', 
        'M 8 25 C 9 21 9 10.3333 9 3 M 4 7 C 5.6667 6.3333 7.3333 4.6667 9 3 C 10 1.6667 11 0.3333 11 -1',
        is_hollow=False),
    'h': Glyph('h', 
        'M 4 0 L 4 4 M 0 4 L 0 2 M 0 0 L 4 0',
        is_hollow=True),
    'd': Glyph('d', 
        'M 20 15 C 17 22 13 26 6 33 M 28 31 C 22 31 16 31 7 32 M 29 33 C 28 31 26 28 25 25 M 14 19 C 12 22 11 24 8 27',
        is_hollow=False),
    'f': Glyph('f', 
        'M 31 -30 C 36.3333 -30 41.6667 -30 47 -30 M 47 -30 L 31 -46 M 39 -38 L 43 -42',
        is_hollow=False),
    'v': Glyph('v', 
        'M 6 13 L 6 -47 M 2 -53 C 6 -51 8 -49 9 -47 M 10 -51 L 59 -51 L 59 10 C 59 13 57 14 54 14 L 51 14',
        is_hollow=True,
        padding=[0.1, 0.1, 0.05, 0.15]),
    'z': Glyph('z', 
        'M 13 0 C 14 -3 16 -1 16 -28 L 40 -28 L 40 -23 L 16 -23 M 29 -28 C 29 -29 29 -29 28 -30',
        is_hollow=True,
        padding=[0.27, 0.02, 0.02, 0.15]),
    't': Glyph('t', 
        'M 62 -102 C 65 -100 69 -94 70 -91 M 62 -86 C 66 -85 68 -83 70 -80 M 64 -54 C 65 -58 67 -62 70 -66',
        is_hollow=False,
        margin=[0.2,0,0.2,0],
        translate=[0, 0])
}

'''

GLYPHS = {
    'r': Glyph('r', 
        'M 45 -69 L 47 -67 L 45 -65 M 47 -69 L 49 -67 L 47 -65 M 49 -69 L 51 -67 L 49 -65',
        is_hollow=False,
        padding=[0.13, 0.15, 0.05, 0.12]),
    's': Glyph('s', 
        'M 0 11 L 0 0 L 11 0 L 11 11 M 11 10 L 0 10',
        is_hollow=True,
        padding=[0.05, 0.05, 0.15, 0.05]),
    'g': Glyph('g', 
        'M 8 25 C 9 21 9 10.3333 9 3 M 4 7 C 5.6667 6.3333 7.3333 4.6667 9 3 C 10 1.6667 11 0.3333 11 -1',
        is_hollow=False),
    'v': Glyph('v', 
        'M 6 13 L 6 -47 M 2 -53 C 6 -51 8 -49 9 -47 M 10 -51 L 59 -51 L 59 10 C 59 13 57 14 54 14 L 51 14',
        is_hollow=True,
        padding=[0.1, 0.1, 0.05, 0.15]),
}
'''
def create_glyph_tree(word):
    random.seed(word)

    comps = []
    for i in range(len(word)):
        if word[i] not in GLYPHS:
            continue
        comps.append(Composition(leaf_glyph=GLYPHS[word[i]]))

    l = len(comps)
    while l > 1:
        #idx = random.randint(0, l - 2)
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

def draw_sentence(sentence, filename, size=400, stroke=5):
    words = sentence.split(' ')
    
    word_trees = []
    for w in words:
        t = create_glyph_tree(w)
        t.calc_constructions()
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

    

draw_sentence('tvgggg', 'out.svg', 200, 5)
