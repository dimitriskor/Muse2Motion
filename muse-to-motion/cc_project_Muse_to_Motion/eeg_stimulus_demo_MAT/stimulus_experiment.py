import time
import random
import os
from psychopy import visual, core, event, sound

# Setup log and record experiment start time
log = []  # To store timestamps
experiment_start_unix = time.time()
print(f" Experiment started at Unix time: {experiment_start_unix}")
log.append(("Experiment Unix Start", experiment_start_unix))

# Setup the window
win = visual.Window(fullscr=False, color="black", units="height")

# Setup clock
global_clock = core.Clock()

# Text stimuli
instruction_text = visual.TextStim(win, text="Get ready...", color="white", height=0.05)
fixation = visual.TextStim(win, text="+", color="white", height=0.1)
resting_text = visual.TextStim(win, text="Resting...", color="white", height=0.05)
music_text = visual.TextStim(win, text="Music playing...\n(Please keep your eyes open)", color="white", height=0.05)

# Math problem generator (more difficult)
def generate_problem():
    start = random.randint(300, 999)
    step = random.randint(10, 50)
    op = random.choice(["+", "-"])
    repeat = 4  # How many steps to show explicitly
    sequence = f"{start}"
    for _ in range(repeat):
        sequence += f" {op} {step}"
    return f"({sequence} ...)"

# Save timestamp
def log_event(event):
    log.append((event, global_clock.getTime()))

# Save log file
def save_log():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file_path = os.path.join(script_dir, "eeg_experiment_timeline_log.txt")
    with open(log_file_path, "w") as f:
        for entry in log:
            f.write(f"{entry[0]}\t{entry[1]:.3f}\n")
    print(f" Log saved to: {log_file_path}")

# Escape key checker (used consistently throughout)
def check_for_escape():
    if "escape" in event.getKeys():
        log_event("Experiment Aborted by User")
        save_log()
        win.close()
        core.quit()

# --- Baseline Rest ---
resting_text.text = "Baseline Rest: Stay still, eyes open"
resting_text.draw()
win.flip()
log_event("Baseline Start")
for _ in range(60):
    core.wait(1)
    check_for_escape()
log_event("Baseline End")

# --- Instruction ---
instruction_text.draw()
win.flip()
log_event("Instruction Start")
core.wait(2)
log_event("Instruction End")
check_for_escape()

# --- Main Task Block (20 trials) ---
for trial in range(20):
    # Task
    problem = generate_problem()
    task_text = visual.TextStim(win, text=problem, color="white", height=0.07)
    task_text.draw()
    win.flip()
    log_event(f"Task Start Trial {trial+1}: {problem}")
    for _ in range(10):
        core.wait(1)
        check_for_escape()
    log_event(f"Task End Trial {trial+1}")

    # Rest
    fixation.draw()
    win.flip()
    log_event(f"Rest Start Trial {trial+1}")
    for _ in range(10):
        core.wait(1)
        check_for_escape()
    log_event(f"Rest End Trial {trial+1}")

# --- Post-task Rest ---
resting_text.text = "Post-task Rest: Eyes open, relax"
resting_text.draw()
win.flip()
log_event("Post-task Rest Start")
for _ in range(60):
    core.wait(1)
    check_for_escape()
log_event("Post-task Rest End")

# --- Music Block ---
music_text.draw()
win.flip()
log_event("Music Start")

# Correct full path to your music file
try:
    relaxing_music = sound.Sound("D:\\Psychopy\\cc_project\\eeg_stimulus_demo\\relaxing_music.wav")
    relaxing_music.play()
except Exception as e:
    print("Music playback failed:", e)

for _ in range(120):
    core.wait(1)
    check_for_escape()
log_event("Music End")

# --- Post-music Rest ---
resting_text.text = "Post-music Rest: Eyes open"
resting_text.draw()
win.flip()
log_event("Post-music Rest Start")
for _ in range(60):
    core.wait(1)
    check_for_escape()
log_event("Post-music Rest End")

# --- End ---
end_text = visual.TextStim(win, text="Session Complete\nThank you!", color="white", height=0.06)
end_text.draw()
win.flip()
log_event("Experiment End")
core.wait(5)

save_log()
win.close()
core.quit()
