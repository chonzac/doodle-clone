import random
import time
import pygame
from PIL import Image

from model import load_checkpoint, predict


WIDTH, HEIGHT = 1000, 650
FPS = 180

CANVAS_RECT = pygame.Rect(40, 120, 600, 480)
UI_X = 700

ROUND_SECONDS = 25
IDLE_GUESS_SECONDS = 0.8
PREDICT_EVERY_SECONDS = 0.25
CONFIDENT_THRESHOLD = 0.80        # 70% threshold
CONFIRM_COUNTDOWN_SECONDS = 5.0   # 5 second countdown

BRUSH_SIZE = 8
INVERT_FOR_MODEL = True

SPRITES_THINK = ["think1.png", "think2.png", "think3.png", "think4.png"]
SPRITE_DONE = "doneguessing.png"
SPRITE_DONE_WRONG = "doneguessing_bad.png"
MUSIC_FILE = "music.mp3"
MODEL_FILE = "doodle_model.pt"
DING_FILE = "ding.mp3"
ERR_FILE  = "err.mp3"

# border animation speed (increase = slower)
BORDER_JITTER_EVERY = 0.25  # seconds
BORDER_SHIFT_PX = 2
BORDER_ROTATE_DEG = 1.5


# -----------------------------
# helping thingamabobs
# -----------------------------
def load_image(path, scale=None):
    img = pygame.image.load(path).convert_alpha()
    if scale is not None:
        img = pygame.transform.smoothscale(img, scale)
    return img

def surface_to_pil_gray(canvas_surf: pygame.Surface) -> Image.Image:
    raw = pygame.image.tostring(canvas_surf, "RGB")
    return Image.frombytes("RGB", canvas_surf.get_size(), raw)

def pick_three_categories(all_classes):
    return random.sample(all_classes, k=min(3, len(all_classes)))

def draw_rotated_panel(screen, rect, offset, angle_deg, draw_contents_fn):
    """
    Draw a white panel + outline + its contents on one surface,
    then rotate/offset the whole thing together so text/sprites move with the box.
    """
    ox, oy = offset

    # Create panel surface the same size as rect (no padding), draw inside it
    panel = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)

    # White fill
    panel.fill((255, 255, 255))

    # Outline
    pygame.draw.rect(panel, (30, 30, 30), pygame.Rect(0, 0, rect.w, rect.h), 2)

    # Contents
    draw_contents_fn(panel)

    # Rotate the whole panel
    rotated = pygame.transform.rotate(panel, angle_deg)
    r = rotated.get_rect(center=(rect.centerx + ox, rect.centery + oy))
    screen.blit(rotated, r.topleft)

def draw_rotated_outline_only(screen, rect, offset, angle_deg):
    """
    border that shifts + slightly rotates
    Used for the canvas border ONLY (we dont rotate the canvas pixels themselves)
    """
    ox, oy = offset
    surf = pygame.Surface((rect.w + 12, rect.h + 12), pygame.SRCALPHA)
    outline_rect = pygame.Rect(6, 6, rect.w, rect.h)
    pygame.draw.rect(surf, (30, 30, 30), outline_rect, 2)

    rotated = pygame.transform.rotate(surf, angle_deg)
    r = rotated.get_rect(center=(rect.centerx + ox, rect.centery + oy))
    screen.blit(rotated, r.topleft)

# ---------- SUPER SMOOTH DRAWING HELPERS ----------
def _midpoint(a, b):
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)

def draw_quadratic_bezier_stamps(surface, p0, p1, p2, radius):
    """
    brush
    """
    dx = p2[0] - p0[0]
    dy = p2[1] - p0[1]
    dist = (dx * dx + dy * dy) ** 0.5
    steps = max(10, int(dist * 0.45))  # higher = smoother (slightly more CPU)

    for i in range(steps + 1):
        t = i / steps
        mt = 1.0 - t
        x = mt * mt * p0[0] + 2 * mt * t * p1[0] + t * t * p2[0]
        y = mt * mt * p0[1] + 2 * mt * t * p1[1] + t * t * p2[1]
        pygame.draw.circle(surface, (0, 0, 0), (int(x), int(y)), radius)


# -----------------------------
# main main main main main
# -----------------------------
def main():
    pygame.init()
    pygame.mixer.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("doodle guessing game!11!1!")
    clock = pygame.time.Clock()

    font_title = pygame.font.SysFont("comicsansms", 32, bold=True)
    font_ui = pygame.font.SysFont("comicsansms", 22)
    font_guess = pygame.font.SysFont("comicsansms", 28, bold=True)

    think_frames = [load_image(p, scale=(220, 220)) for p in SPRITES_THINK]
    done_frame = load_image(SPRITE_DONE, scale=(220, 220))
    done_frame_wrong = load_image(SPRITE_DONE_WRONG, scale=(220, 220))

    # music
    try:
        pygame.mixer.music.load(MUSIC_FILE)
        pygame.mixer.music.play(-1)
    except Exception as e:
        print("Music load/play failed:", e)

    # sound effects
    try:
        ding_snd = pygame.mixer.Sound(DING_FILE)
    except Exception as e:
        ding_snd = None
        print("ding.mp3 failed:", e)

    try:
        err_snd = pygame.mixer.Sound(ERR_FILE)
    except Exception as e:
        err_snd = None
        print("err.mp3 failed:", e)

    result_sound_played = False

    model, classes = load_checkpoint(MODEL_FILE)

    state = "choose"
    options = pick_three_categories(classes)
    chosen = None

    canvas = pygame.Surface((CANVAS_RECT.w, CANVAS_RECT.h))
    canvas.fill((255, 255, 255))

    drawing = False
    last_draw_time = 0.0
    last_predict_time = 0.0
    round_start_time = 0.0

    locked_guess = None
    locked_conf = 0.0

    live_guess = None
    live_conf = 0.0
    is_confident = False

    confirm_start_time = None

    think_idx = 0
    think_last_swap = 0.0
    THINK_SWAP_EVERY = 0.18

    # ---- smooth drawing stroke state ----
    stroke_points = []

    # ---- border state: shift + rotation updates every BORDER_JITTER_EVERY ----
    border_time = 0.0
    border_canvas_offset = (0, 0)
    border_ui_offset = (0, 0)
    border_btn_offset = (0, 0)

    border_canvas_angle = 0.0
    border_ui_angle = 0.0
    border_btn_angle = 0.0

    def reset_round():
        nonlocal options, chosen, drawing, last_draw_time, last_predict_time
        nonlocal round_start_time, locked_guess, locked_conf, live_guess, live_conf
        nonlocal is_confident, result_sound_played, confirm_start_time, stroke_points

        options = pick_three_categories(classes)
        chosen = None
        canvas.fill((255, 255, 255))
        drawing = False
        last_draw_time = 0.0
        last_predict_time = 0.0
        round_start_time = 0.0
        locked_guess = None
        locked_conf = 0.0
        live_guess = None
        live_conf = 0.0
        is_confident = False
        result_sound_played = False
        confirm_start_time = None
        stroke_points = []

    def option_rect(i):
        return pygame.Rect(80, 250 + i * 85, 520, 60)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        now = time.time()

        # ----------------- events -----------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    state = "choose"
                    reset_round()

            if state == "choose":
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    for i, opt in enumerate(options):
                        if option_rect(i).collidepoint(mx, my):
                            chosen = opt
                            state = "draw"
                            round_start_time = now
                            canvas.fill((255, 255, 255))
                            live_guess, live_conf = None, 0.0
                            is_confident = False
                            confirm_start_time = None
                            stroke_points = []

            elif state == "draw":
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if CANVAS_RECT.collidepoint(event.pos):
                        drawing = True
                        last_draw_time = now
                        mx, my = event.pos
                        lx = mx - CANVAS_RECT.x
                        ly = my - CANVAS_RECT.y
                        stroke_points = [(lx, ly)]
                        pygame.draw.circle(canvas, (0, 0, 0), (lx, ly), BRUSH_SIZE)

                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    drawing = False
                    last_draw_time = now
                    stroke_points = []

                if event.type == pygame.MOUSEMOTION and drawing:
                    mx, my = event.pos
                    if CANVAS_RECT.collidepoint(mx, my):
                        lx = mx - CANVAS_RECT.x
                        ly = my - CANVAS_RECT.y
                        stroke_points.append((lx, ly))

                        if len(stroke_points) >= 3:
                            a, b, c = stroke_points[-3], stroke_points[-2], stroke_points[-1]
                            m1 = _midpoint(a, b)
                            m2 = _midpoint(b, c)
                            draw_quadratic_bezier_stamps(canvas, m1, b, m2, BRUSH_SIZE)
                        else:
                            pygame.draw.circle(canvas, (0, 0, 0), (int(lx), int(ly)), BRUSH_SIZE)

                        last_draw_time = now

            elif state == "result":
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    state = "choose"
                    reset_round()

        # logic updates -----------------------------------------------------------
        if state == "draw":
            elapsed = now - round_start_time
            remaining = max(0.0, ROUND_SECONDS - elapsed)

            if remaining <= 0.0:
                pil_img = surface_to_pil_gray(canvas)
                top = predict(model, classes, pil_img, invert=INVERT_FOR_MODEL, topk=1)[0]
                locked_guess, locked_conf = top
                state = "result"
                live_guess, live_conf = locked_guess, locked_conf
                is_confident = True

                if not result_sound_played:
                    result_sound_played = True
                    if locked_guess == chosen:
                        if err_snd: err_snd.play()
                    else:
                        if ding_snd: ding_snd.play()

            else:
                idle_for = now - last_draw_time if last_draw_time > 0 else 0
                should_guess = (last_draw_time > 0) and (idle_for >= IDLE_GUESS_SECONDS)

                if should_guess and (now - last_predict_time >= PREDICT_EVERY_SECONDS):
                    last_predict_time = now
                    pil_img = surface_to_pil_gray(canvas)
                    top = predict(model, classes, pil_img, invert=INVERT_FOR_MODEL, topk=1)[0]
                    live_guess, live_conf = top
                    is_confident = (live_conf >= CONFIDENT_THRESHOLD)

                    if is_confident:
                        if confirm_start_time is None:
                            confirm_start_time = now
                        elif (now - confirm_start_time) >= CONFIRM_COUNTDOWN_SECONDS:
                            locked_guess, locked_conf = live_guess, live_conf
                            state = "result"
                    else:
                        confirm_start_time = None

        # thinking animation
        if now - think_last_swap >= THINK_SWAP_EVERY:
            think_last_swap = now
            think_idx = (think_idx + 1) % len(think_frames)

        # update border offset/rotation only every BORDER_JITTER_EVERY seconds
        if now - border_time >= BORDER_JITTER_EVERY:
            border_time = now
            border_canvas_offset = (
                random.randint(-BORDER_SHIFT_PX, BORDER_SHIFT_PX),
                random.randint(-BORDER_SHIFT_PX, BORDER_SHIFT_PX),
            )
            border_ui_offset = (
                random.randint(-BORDER_SHIFT_PX, BORDER_SHIFT_PX),
                random.randint(-BORDER_SHIFT_PX, BORDER_SHIFT_PX),
            )
            border_btn_offset = (
                random.randint(-BORDER_SHIFT_PX, BORDER_SHIFT_PX),
                random.randint(-BORDER_SHIFT_PX, BORDER_SHIFT_PX),
            )

            border_canvas_angle = random.uniform(-BORDER_ROTATE_DEG, BORDER_ROTATE_DEG)
            border_ui_angle = random.uniform(-BORDER_ROTATE_DEG, BORDER_ROTATE_DEG)
            border_btn_angle = random.uniform(-BORDER_ROTATE_DEG, BORDER_ROTATE_DEG)

        # ----------------- render -----------------
        screen.fill((245, 242, 235))

        title = font_title.render("doodle guessing game!1!!1! (by chonza)", True, (20, 20, 20))
        subtitle = font_title.render("join programming club!", True, (100, 100, 100))
        screen.blit(title, (40, 30))
        screen.blit(subtitle, (20, 60))
        # canvas
        pygame.draw.rect(screen, (255, 255, 255), CANVAS_RECT)
        screen.blit(canvas, (CANVAS_RECT.x, CANVAS_RECT.y))
        draw_rotated_outline_only(screen, CANVAS_RECT, border_canvas_offset, border_canvas_angle)

        # monkey live guess
        ui_rect = pygame.Rect(UI_X - 20, 120, 280, 480)

        def draw_ui_contents(panel):
            # instructions
            panel.blit(font_ui.render("r to restart", True, (50, 50, 50)), (20, 10))
            panel.blit(font_ui.render("escape to quit", True, (50, 50, 50)), (20, 40))

            if state == "choose":
                panel.blit(font_ui.render("pick a category :p", True, (20, 20, 20)), (20, 90))
                panel.blit(think_frames[think_idx], (30, 130))

            elif state == "draw":
                panel.blit(font_ui.render(f"Draw: {chosen}", True, (20, 20, 20)), (20, 90))

                # normal timer replaced with lock-in countdown when confirm_start_time exists
                if confirm_start_time is None:
                    remaining_int = max(0, int(ROUND_SECONDS - (time.time() - round_start_time)))
                    panel.blit(font_ui.render(f"Time: {remaining_int:02d}s", True, (20, 20, 20)), (20, 120))
                else:
                    secs_left = max(
                        0,
                        int(CONFIRM_COUNTDOWN_SECONDS - (time.time() - confirm_start_time) + 0.99)
                    )
                    panel.blit(font_guess.render(f"LOCKING IN: {secs_left}", True, (0, 140, 0)), (20, 120))

                # display for the guesser monkey
                if live_guess is None:
                    guess_text = "…"
                    color = (200, 0, 0)
                    confident = False
                else:
                    confident = (live_conf >= CONFIDENT_THRESHOLD)
                    color = (0, 140, 0) if confident else (200, 0, 0)
                    guess_text = f"{live_guess}" + ("" if confident else "…")

                conf_line = f"{int(live_conf * 100)}%" if live_guess else ""

                panel.blit(font_guess.render(guess_text, True, color), (20, 170))
                panel.blit(font_ui.render(conf_line, True, color), (20, 210))

                # sprite based on confidence
                if live_guess is not None and confident:
                    panel.blit(done_frame, (30, 280))
                else:
                    panel.blit(think_frames[think_idx], (30, 280))

            elif state == "result":
                panel.blit(font_ui.render("the final guess....", True, (20, 20, 20)), (20, 90))

                color = (0, 140, 0) if live_guess == chosen else (200, 0, 0)
                panel.blit(font_guess.render(f"{live_guess} ({int(live_conf*100)}%)", True, color), (20, 130))
                panel.blit(font_ui.render(f"right answer {chosen}", True, (20, 20, 20)), (20, 175))

                if live_guess == chosen:
                    panel.blit(done_frame, (30, 220))
                else:
                    panel.blit(done_frame_wrong, (30, 220))

                panel.blit(font_ui.render("click to play again!11!", True, (20, 20, 20)), (20, 440))

        draw_rotated_panel(screen, ui_rect, border_ui_offset, border_ui_angle, draw_ui_contents)

        # el buttons
        if state == "draw":
            tip = font_ui.render("stop drawing (like a second) to let the ai guess", True, (120, 20, 40))
            screen.blit(tip, (UI_X - 400, 600))

        
        if state == "choose":
            for i, opt in enumerate(options):
                r = option_rect(i)

                def make_btn_contents(txt):
                    def _draw(panel):
                        t = font_guess.render(txt, True, (20, 20, 20))
                        panel.blit(t, (20, (panel.get_height() - t.get_height()) // 2))
                    return _draw

                draw_rotated_panel(screen, r, border_btn_offset, border_btn_angle, make_btn_contents(opt))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
