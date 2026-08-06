"""Interactive berth-slot rectangle picker for Jupyter (drag + resize boxes)."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, RectangleSelector


def format_berth_slots_python(slots: list[dict], var_name: str = "MUAJJIZ_BERTH_SLOTS") -> str:
    lines = [f"{var_name} = ["]
    for s in slots:
        x1, y1, x2, y2 = s["box"]
        w, h = x2 - x1, y2 - y1
        lines.append(
            f'    {{"name": "{s["name"]}", "box": [{x1}, {y1}, {x2}, {y2}]}},  # {w}x{h} px'
        )
    lines.append("]")
    return "\n".join(lines)


def pick_berth_slots_interactive(
    scene,
    existing_slots: list[dict] | None = None,
    slot_names: list[str] | None = None,
    var_name: str = "MUAJJIZ_BERTH_SLOTS",
):
    """
    Draw berth boxes on a satellite scene.

    Requires an interactive matplotlib backend, e.g. run once in the notebook:
        %matplotlib widget
    (pip install ipympl)

    Controls:
      - Click-drag to create a box; drag corners/edges to move or resize it
      - **Accept box** — save the current rectangle and start the next berth
      - **Undo** — remove the last saved box
      - Close the figure window when finished

    Returns the list of {"name", "box": [x1, y1, x2, y2]} dicts.
    """
    slots = [dict(s) for s in (existing_slots or [])]
    names = list(slot_names or [])
    h_img, w_img = scene.shape[:2]
    patches: list[Rectangle] = []

    fig, ax = plt.subplots(figsize=(12, 12))
    plt.subplots_adjust(bottom=0.08)
    ax.imshow(scene)
    ax.set_xlim(0, w_img)
    ax.set_ylim(h_img, 0)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")

    status = fig.text(0.5, 0.01, "", ha="center", fontsize=10, family="monospace")
    selector: RectangleSelector | None = None

    def next_name() -> str:
        i = len(slots)
        return names[i] if i < len(names) else f"berth_{i + 1}"

    def redraw_saved():
        for p in patches:
            p.remove()
        patches.clear()
        for s in slots:
            x1, y1, x2, y2 = s["box"]
            bw, bh = x2 - x1, y2 - y1
            rect = Rectangle(
                (x1, y1), bw, bh, fill=False, edgecolor="yellow", linewidth=2, linestyle="--"
            )
            ax.add_patch(rect)
            ax.text(x1 + 2, y1 + 10, s["name"], color="yellow", fontsize=10, weight="bold")
        fig.canvas.draw_idle()

    def print_slots():
        text = format_berth_slots_python(slots, var_name=var_name)
        print("\n" + text + "\n")
        status.set_text(f"Saved {len(slots)} box(es) — copy the list above into AOI_CONFIG")

    def on_select(eclick, erelease):
        if selector is None or not hasattr(selector, "extents"):
            return
        x1, x2, y1, y2 = selector.extents
        bw, bh = abs(x2 - x1), abs(y2 - y1)
        ax.set_title(
            f"Draw box for {next_name()} — drag corners to resize ({bw:.0f}x{bh:.0f} px)",
            fontsize=11,
        )
        fig.canvas.draw_idle()

    def new_selector():
        nonlocal selector
        if selector is not None:
            selector.set_active(False)
        selector = RectangleSelector(
            ax,
            on_select,
            useblit=True,
            button=[1],
            minspanx=8,
            minspany=8,
            spancoords="pixels",
            interactive=True,
            props=dict(facecolor="lime", edgecolor="lime", alpha=0.25, linewidth=2),
        )
        ax.set_title(f"Click-drag a box for {next_name()}, then press Accept box", fontsize=11)

    def accept(_event):
        if selector is None or not hasattr(selector, "extents"):
            print("Draw a rectangle first (click-drag on the image).")
            return
        x1, x2, y1, y2 = selector.extents
        x1, x2 = sorted([max(0, x1), min(w_img, x2)])
        y1, y2 = sorted([max(0, y1), min(h_img, y2)])
        bw, bh = int(round(x2 - x1)), int(round(y2 - y1))
        if bw < 8 or bh < 8:
            print("Box too small — drag a larger rectangle.")
            return
        name = next_name()
        box = [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
        slots.append({"name": name, "box": box})
        print(f"Accepted {name}: box={box}  size={bw}x{bh} px")
        redraw_saved()
        print_slots()
        new_selector()

    def undo(_event):
        if not slots:
            print("Nothing to undo.")
            return
        removed = slots.pop()
        print(f"Removed {removed['name']}")
        redraw_saved()
        print_slots()
        new_selector()

    ax_accept = fig.add_axes([0.55, 0.02, 0.14, 0.045])
    ax_undo = fig.add_axes([0.38, 0.02, 0.14, 0.045])
    ax_done = fig.add_axes([0.72, 0.02, 0.14, 0.045])
    Button(ax_accept, "Accept box").on_clicked(accept)
    Button(ax_undo, "Undo").on_clicked(undo)
    Button(ax_done, "Done").on_clicked(lambda _e: plt.close(fig))

    redraw_saved()
    new_selector()
    print(
        "Interactive picker open.\n"
        "  1. Click-drag to draw a berth box\n"
        "  2. Drag corners/edges to adjust size and position\n"
        "  3. Click 'Accept box' for each berth\n"
        "  4. Click 'Done' or close the window when finished\n"
        f"Image size: {w_img} x {h_img} px (box coords are [x1, y1, x2, y2])"
    )
    if existing_slots:
        print_slots()

    plt.show()
    return slots


def pick_berth_slots_tk(
    scene,
    existing_slots: list[dict] | None = None,
    slot_names: list[str] | None = None,
    var_name: str = "MUAJJIZ_BERTH_SLOTS",
) -> list[dict]:
    """
    Tkinter berth picker — works in Cursor/VS Code without %matplotlib widget.

    Click-drag a box, press Enter to accept, u to undo, q to quit.
    To resize: draw again (undo first if needed) or edit the printed coords.
    """
    import tkinter as tk
    from PIL import Image, ImageTk

    h, w = scene.shape[:2]
    slots = [dict(s) for s in (existing_slots or [])]
    names = list(slot_names or [f"berth_{i+1}" for i in range(3)])

    scale = min(1.0, 950 / max(h, w))
    dw, dh = max(1, int(w * scale)), max(1, int(h * scale))

    root = tk.Tk()
    root.title("Berth picker — drag box | Enter=accept | u=undo | q=quit")

    pil = Image.fromarray(scene)
    if scale != 1.0:
        pil = pil.resize((dw, dh), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(pil)
    root._photo = photo  # keep reference so canvas doesn't lose the image

    status = tk.StringVar(value="Drag a rectangle around a berth, then press Enter")
    tk.Label(root, textvariable=status, font=("Consolas", 10)).pack(fill=tk.X, padx=8, pady=4)

    canvas = tk.Canvas(root, width=dw, height=dh, cursor="cross")
    canvas.pack(padx=8, pady=4)
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    drag_start = None
    rubber_id = None
    pending = None
    saved_ids: list[int] = []

    def to_img(x: float, y: float) -> tuple[int, int]:
        return int(round(x / scale)), int(round(y / scale))

    def next_name() -> str:
        i = len(slots)
        return names[i] if i < len(names) else f"berth_{i + 1}"

    def draw_saved():
        nonlocal saved_ids
        for sid in saved_ids:
            canvas.delete(sid)
        saved_ids.clear()
        for s in slots:
            x1, y1, x2, y2 = s["box"]
            rx1, ry1, rx2, ry2 = x1 * scale, y1 * scale, x2 * scale, y2 * scale
            rid = canvas.create_rectangle(rx1, ry1, rx2, ry2, outline="yellow", width=2, dash=(4, 2))
            tid = canvas.create_text(rx1 + 4, ry1 + 4, anchor=tk.NW, text=s["name"], fill="yellow")
            saved_ids.extend([rid, tid])

    def on_press(event):
        nonlocal drag_start, rubber_id, pending
        drag_start = (event.x, event.y)
        pending = None
        if rubber_id is not None:
            canvas.delete(rubber_id)
        rubber_id = canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="lime", width=2)

    def on_drag(event):
        if drag_start is None or rubber_id is None:
            return
        canvas.coords(rubber_id, drag_start[0], drag_start[1], event.x, event.y)
        x1, y1 = to_img(min(drag_start[0], event.x), min(drag_start[1], event.y))
        x2, y2 = to_img(max(drag_start[0], event.x), max(drag_start[1], event.y))
        status.set(f"Pending {next_name()}: [{x1},{y1},{x2},{y2}]  size={x2-x1}x{y2-y1} px — Enter=accept")

    def on_release(event):
        nonlocal pending, drag_start
        if drag_start is None:
            return
        x1, y1 = to_img(min(drag_start[0], event.x), min(drag_start[1], event.y))
        x2, y2 = to_img(max(drag_start[0], event.x), max(drag_start[1], event.y))
        pending = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
        x1, y1, x2, y2 = pending
        status.set(f"Pending {next_name()}: [{x1},{y1},{x2},{y2}]  size={x2-x1}x{y2-y1} px — Enter=accept")
        drag_start = None

    def accept(_event=None):
        nonlocal pending, rubber_id
        if pending is None:
            status.set("Draw a box first (click-drag), then press Enter")
            return
        x1, y1, x2, y2 = pending
        if x2 - x1 < 5 or y2 - y1 < 5:
            status.set("Box too small — drag a larger rectangle")
            return
        name = next_name()
        box = [x1, y1, x2, y2]
        slots.append({"name": name, "box": box})
        pending = None
        if rubber_id is not None:
            canvas.delete(rubber_id)
            rubber_id = None
        draw_saved()
        print(f"Accepted {name}: box={box}  size={x2-x1}x{y2-y1} px")
        print(format_berth_slots_python(slots, var_name=var_name))
        status.set(f"Saved {name}. Drag next box or press q to quit.")

    def undo(_event=None):
        nonlocal pending, rubber_id
        if slots:
            removed = slots.pop()
            print(f"Removed {removed['name']}")
            draw_saved()
            print(format_berth_slots_python(slots, var_name=var_name))
        pending = None
        if rubber_id is not None:
            canvas.delete(rubber_id)
            rubber_id = None
        status.set("Undo done. Drag a new box.")

    def quit_app(_event=None):
        root.quit()
        root.destroy()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    root.bind("<Return>", accept)
    root.bind("u", undo)
    root.bind("q", quit_app)
    root.protocol("WM_DELETE_WINDOW", quit_app)

    draw_saved()
    print(f"Image {w}x{h} px (display scaled to {dw}x{dh})")
    print("Drag box → Enter to accept → u undo → q quit")
    root.mainloop()
    return slots
