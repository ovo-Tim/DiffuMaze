import torch
from PIL import Image

ROUTE_COLORS = [
    (220, 50, 50),
    (50, 180, 50),
    (50, 80, 220),
    (220, 180, 30),
    (180, 50, 180),
    (50, 180, 180),
    (200, 120, 50),
    (120, 50, 200),
    (50, 200, 100),
    (200, 50, 120),
]


def assign_route_labels(puzzle_1chk, solution):
    """
    Assign each solution pixel to a route by flood-filling from checkpoints.

    puzzle_1chk: (g1, h, w)  - ch0=walls, ch1..g=checkpoints
    solution:    (h, w)      - binary solution path [0,1] or continuous [0,1]
    Returns:     (h, w) long  - -1 = not solution, 0..g-1 = route index
    """
    g1, h, w = puzzle_1chk.shape
    g = g1 - 1
    walls = puzzle_1chk[0]
    checkpoints = puzzle_1chk[1:]

    assignment = torch.full((h, w), -1, dtype=torch.long)
    mask = solution > 0.5

    for r in range(g):
        chk = checkpoints[r]
        positions = torch.nonzero(chk)
        starts = [(int(p[0]), int(p[1])) for p in positions if mask[int(p[0]), int(p[1])]]

        for sy, sx in starts:
            if assignment[sy, sx] != -1:
                continue
            stack = [(sy, sx)]
            while stack:
                y, x = stack.pop()
                if not (0 <= y < h and 0 <= x < w):
                    continue
                if not mask[y, x] or assignment[y, x] != -1:
                    continue
                assignment[y, x] = r
                stack.append((y - 1, x))
                stack.append((y + 1, x))
                stack.append((y, x - 1))
                stack.append((y, x + 1))

    return assignment


def render_colored(puzzle_1chk, solution, scale=8):
    g1, h, w = puzzle_1chk.shape
    walls = puzzle_1chk[0]
    assignment = assign_route_labels(puzzle_1chk, solution)

    img = Image.new("RGB", (w * scale, h * scale))
    px = img.load()

    for y in range(h):
        for x in range(w):
            if walls[y, x] > 0.5:
                c = (0, 0, 0)
            elif assignment[y, x] >= 0:
                c = ROUTE_COLORS[assignment[y, x] % len(ROUTE_COLORS)]
            else:
                c = (128, 128, 128)
            for dy in range(scale):
                for dx in range(scale):
                    px[x * scale + dx, y * scale + dy] = c

    for g in range(1, g1):
        chk = puzzle_1chk[g]
        for y in range(h):
            for x in range(w):
                if chk[y, x] > 0.5:
                    for dy in range(scale):
                        for dx in range(scale):
                            px[x * scale + dx, y * scale + dy] = (255, 255, 255)

    return img


def render_heatmap(puzzle_1chk, values, scale=8):
    h, w = values.shape
    walls = puzzle_1chk[0]
    vals = values.cpu().numpy() if isinstance(values, torch.Tensor) else values

    img = Image.new("RGB", (w * scale, h * scale))
    px = img.load()

    for y in range(h):
        for x in range(w):
            if walls[y, x] > 0.5:
                c = (0, 0, 0)
            else:
                v = max(-1.0, min(1.0, float(vals[y, x])))
                gray = int((v + 1.0) / 2.0 * 255.0)
                c = (gray, gray, gray)
            for dy in range(scale):
                for dx in range(scale):
                    px[x * scale + dx, y * scale + dy] = c

    return img


def concat_images(img1, img2, gap=10, label_left="", label_right=""):
    from PIL import ImageDraw
    w = img1.width + img2.width + gap
    h = max(img1.height, img2.height)
    canvas = Image.new("RGB", (w, h), (40, 40, 40))
    canvas.paste(img1, (0, 0))
    canvas.paste(img2, (img1.width + gap, 0))
    if label_left or label_right:
        draw = ImageDraw.Draw(canvas)
        if label_left:
            draw.text((4, 4), label_left, fill=(255, 255, 255))
        if label_right:
            draw.text((img1.width + gap + 4, 4), label_right, fill=(255, 255, 255))
    return canvas