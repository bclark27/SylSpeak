import svgwrite
from enum import Enum
import random
import math




class LineType(Enum):
    S = 1
    SS = 2
    BS = 3
    SB = 4
    BB = 5
    B = 6

SMALL_STROKE = 5
BIG_STROKE = 10
STROKE_GAP = 3
RING_GAP = 3

def draw_circ(dwg, group, x, y, inner_r, line_type, fill):

  if line_type == LineType.S:
    group.add(dwg.circle(center=(x, y), r=inner_r + (SMALL_STROKE / 2), fill=fill, stroke='black', stroke_width=SMALL_STROKE))
    return
  elif line_type == LineType.B:
    group.add(dwg.circle(center=(x, y), r=inner_r + (BIG_STROKE / 2), fill=fill, stroke='black', stroke_width=BIG_STROKE))
    return


  a = 0
  b = 0

  if line_type == LineType.SS:
    a = SMALL_STROKE
    b = SMALL_STROKE
  elif line_type == LineType.BS:
    a = BIG_STROKE
    b = SMALL_STROKE
  elif line_type == LineType.SB:
    a = SMALL_STROKE
    b = BIG_STROKE
  elif line_type == LineType.BB:
    a = BIG_STROKE
    b = BIG_STROKE

  a_s = a
  a_r = inner_r + (a_s / 2)

  b_s = b
  b_r = inner_r + a_s + STROKE_GAP + (b_s / 2)
  group.add(dwg.circle(center=(x, y), r=b_r, fill=fill, stroke='black', stroke_width=b_s))
  group.add(dwg.circle(center=(x, y), r=a_r, fill=fill, stroke='black', stroke_width=a_s))

def mask_circ(dwg, group, x, y, r, fill):
    L = 1000

    # Combined path: big rectangle + inner circle hole
    path_data = (
        f"M{-L},{-L} H{L} V{L} H{-L} Z "
        f"M{x - r},{y} "
        f"a{r},{r} 0 1,0 {2*r},0 "
        f"a{r},{r} 0 1,0 {-2*r},0 Z"
    )

    path = dwg.path(d=path_data, fill=fill, fill_rule="evenodd")
    group.add(path)


def get_total_line_thickness(line_type):
  if line_type == LineType.S:
    return SMALL_STROKE
  elif line_type == LineType.SS:
    return SMALL_STROKE + SMALL_STROKE + STROKE_GAP
  elif line_type == LineType.BS:
    return BIG_STROKE + SMALL_STROKE + STROKE_GAP
  elif line_type == LineType.SB:
    return BIG_STROKE + SMALL_STROKE + STROKE_GAP
  elif line_type == LineType.BB:
    return BIG_STROKE + BIG_STROKE + STROKE_GAP
  elif line_type == LineType.B:
    return BIG_STROKE
  return 0


class RingEle:

  line_type = None
  my_arc = 0
  my_max_r = 0
  my_angle = 0

  def __init__(self, line_type):
    self.line_type = line_type

  def clone(self):
    return RingEle(self.line_type)

  def get_max_r(self):
    return self.my_max_r

  def get_my_arc(self):
    return self.my_arc

  def get_my_angle(self):
    return self.my_angle

  def draw(self, dwg, group, arc, angle, all_ele, my_idx):
    self.my_arc = arc
    self.my_angle = angle

    # find the max of all the prev ele
    max_r = 0
    for e in all_ele:
      max_r = max(max_r, e.get_max_r())
    max_r += STROKE_GAP


    min_r = 0
    for i in range(my_idx - 1, -1, -1):
      if isinstance(all_ele[i], RingEle):
        min_r = all_ele[i].get_max_r()
        break
    min_r += RING_GAP


    # now need to go out for the gap
    my_inner_radius = random.uniform(min_r, max_r)
    my_outer_radius = my_inner_radius + get_total_line_thickness(self.line_type)

    # the outer radius is now my max radius
    self.my_max_r = my_outer_radius

    # ok now draw the ring
    draw_circ(dwg, group, 0, 0, my_inner_radius, self.line_type, 'none')
    mask_circ(dwg, group, 0, 0, my_outer_radius, 'white')


class GlyphType(Enum):
    INSIDE = 1
    MID = 2
    ON = 3
    ABOVE = 4

class GlyphEle:

  line_type = None
  glyph_type = None
  my_arc = 0
  my_max_r = 0
  my_angle = 0

  def __init__(self, line_type, glyph_type):
    self.glyph_type = glyph_type
    self.line_type = line_type

  def clone(self):
    return GlyphEle(self.line_type, self.glyph_type)

  def get_max_r(self):
    return self.my_max_r

  def get_my_arc(self):
    return self.my_arc

  def get_my_angle(self):
    return self.arc

  def draw(self, dwg, group, arc, angle, all_ele, my_idx):
    self.my_arc = arc
    self.my_angle = angle

    # first get the radius of the last ring

    ring_r = 0
    for i in range(my_idx - 1, -1, -1):
      if isinstance(all_ele[i], RingEle):
        ring_r = all_ele[i].get_max_r()
        break

    # ok we now need to find the max radius we can do using our arc length given
    half_arc_rad = math.radians(arc / 2)
    max_radius = 2 * ring_r * math.sin(half_arc_rad / 2)

    # now we donw want to go crazy so we just do the min here to get a smaller cap
    glyph_outer_radius = min(max_radius, 30)

    # now we need to get the inner radius, so subtract the line thickness
    glyph_inner_radius = glyph_outer_radius - get_total_line_thickness(self.line_type)

    # ok and now depending on which type, adjust the dist we draw the circle at
    glyph_dist = 0
    if self.glyph_type == GlyphType.MID:
      glyph_dist = ring_r
    elif self.glyph_type == GlyphType.ON:
      glyph_dist = ring_r + glyph_outer_radius
    elif self.glyph_type == GlyphType.ABOVE:
      glyph_dist = ring_r + glyph_outer_radius + RING_GAP


    self.my_max_r = glyph_dist + glyph_outer_radius

    # ok we now can draw the thing at coords
    angle_rad = math.radians(angle)
    x = glyph_dist * math.cos(angle_rad)
    y = glyph_dist * math.sin(angle_rad)

    draw_circ(dwg, group, x, y, glyph_inner_radius, self.line_type, 'white')





letter_to_ele = {
  'a': RingEle(LineType.S),
  'e': RingEle(LineType.SS),
  'i': RingEle(LineType.SB),
  'o': RingEle(LineType.BS),
  'u': RingEle(LineType.BB),
  'y': RingEle(LineType.B),

  'q': GlyphEle(LineType.S, GlyphType.MID),
  'w': GlyphEle(LineType.S, GlyphType.ON),
  'r': GlyphEle(LineType.S, GlyphType.ABOVE),
  't': GlyphEle(LineType.SS, GlyphType.MID),
  'p': GlyphEle(LineType.SS, GlyphType.ON),
  's': GlyphEle(LineType.SS, GlyphType.ABOVE),
  'd': GlyphEle(LineType.BS, GlyphType.ON),
  'f': GlyphEle(LineType.BS, GlyphType.MID),
  'g': GlyphEle(LineType.BS, GlyphType.ABOVE),
  'h': GlyphEle(LineType.SB, GlyphType.ON),
  'j': GlyphEle(LineType.SB, GlyphType.MID),
  'k': GlyphEle(LineType.SB, GlyphType.ABOVE),
  'l': GlyphEle(LineType.BB, GlyphType.ON),
  'z': GlyphEle(LineType.BB, GlyphType.MID),
  'x': GlyphEle(LineType.BB, GlyphType.ABOVE),
  'c': GlyphEle(LineType.B, GlyphType.ON),
  'v': GlyphEle(LineType.B, GlyphType.MID),
  'b': GlyphEle(LineType.B, GlyphType.ABOVE),
  'n': GlyphEle(LineType.SS, GlyphType.ON),
  'm': GlyphEle(LineType.SS, GlyphType.MID),
}

def gen_script(word):

  if word[0] not in 'aeiou':
    word = 'e' + word
  if word[-1] not in 'aeiou':
    word += 'u'

  random.seed(word)

  # first create the initial set of elements
  elements = []
  for c in word:
    if c in letter_to_ele:
      elements.append(letter_to_ele[c].clone())

  # check if there is a rin element first. if not, add one
  if not isinstance(elements[0], RingEle):
    elements.insert(0, RingEle(LineType.S))

  # pre plan by checking if there are too many glyphs in a row to stack on the circle
  g_count = 0
  l = len(elements)
  for i in range(l - 1, -1, -1):
    if not isinstance(elements[i], RingEle):
      g_count += 1
    else:
      g_count = 0
      continue

    if g_count > 2:
      elements.insert(i + 1, RingEle(LineType.S))
      g_count = 1

  # Create an SVG drawing (500x500 px)
  canvas_width = 1000
  dwg = svgwrite.Drawing("circle_example.svg", size=(f"{canvas_width}px", f"{canvas_width}px"))
  group = dwg.g()
  for i in range(len(elements)):
    elements[i].draw(dwg, group, 60, random.randint(0,360), elements, i)

  # full_radius = elements[-1].get_max_r()
  # img_w = (full_radius + 20) * 2

  group.translate(canvas_width / 2, canvas_width / 2)
  # dwg['width'] = f'{img_w}px'
  # dwg['height'] = f'{img_w}px'
  dwg.add(group)

  # Save the file
  dwg.save()

def main():
  word = "this is a computer"
  gen_script(word)


main()
