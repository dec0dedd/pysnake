import imageio.v2 as imageio
import os

frame_dir = "frames"
gif_path = "gifs/ppo.gif"

frame_files = sorted(os.listdir(frame_dir))
frames = [imageio.imread(os.path.join(frame_dir, f)) for f in frame_files]
imageio.mimsave(gif_path, frames, duration=0.05)
