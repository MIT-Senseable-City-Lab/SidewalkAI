"""
Generates a .jpg frame sequence from a given video.
"""
import argparse
import cv2
import os


def prompt_user(prompt):
    """Makes sure the user wants to continue."""
    response = input(f"{prompt} (y/n): ")
    if response != "y":
        print("Exiting.")
        exit()


def main():
    # Parse all arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True,
                        help="The video to generate the sequence from")
    parser.add_argument('--output', type=str, required=True,
                        help="The output directory to save the sequence to")
    parser.add_argument('--every', type=int, default=1,
                        help="Save every x frames")
    parser.add_argument('--start_at', type=int, default=0,
                        help="The start time in seconds")
    parser.add_argument('--end_at', type=int, default=None,
                        help="The end time in seconds")
    parser.add_argument('--scale', type=float, default=1.0,
                        help="The scale to resize the frames to")
    args = parser.parse_args()

    # Create output directory
    if os.path.isdir(args.output):
        print(f"Warning: Output directory {args.output} already exists!")
        prompt_user("Overwrite?")
        print("Deleting old directory...")
        os.system(f"rm -rf {args.output}")
    else:
        print(f"Creating output directory: {args.output}")
    os.makedirs(args.output)
    
    # Get video properties
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.end_at is None:
        args.end_at = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first_frame = int(args.start_at * fps)
    last_frame = int(args.end_at * fps) if args.end_at else total_frames
    resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    scaled_res = (int(resolution[0] * args.scale), int(resolution[1] * args.scale))

    # Display info
    print(f"\nVideo length: {total_frames / fps:.1f} seconds")
    print(f"Portion to save: {args.start_at:.1f} to {args.end_at:.1f} seconds")
    print(f"Saving every {args.every} frame(s).")
    print(f"Video FPS: {fps:.1f}  |  Adjusted FPS: {fps / args.every:.1f}")
    print(f"Video resolution: {resolution[0]}x{resolution[1]}  |  Scaled resolution: {scaled_res[0]}x{scaled_res[1]}")

    print(f"\nNumber of frames to save: {(last_frame - first_frame + 1) / args.every:.0f}")
    prompt_user("Continue?")

    # Save frames
    print("Saving frames...")
    for i in range(first_frame, last_frame, args.every):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame {i}")
            break
        frame = cv2.resize(frame, scaled_res)
        # File name is 0-padded
        id = str(i).zfill(len(str(last_frame)))
        cv2.imwrite(f"{args.output}/{id}.jpg", frame)
        print(f"Progress: {(i-first_frame)*100 // (last_frame - first_frame)}%", end='\n')
    print("Done!")


if __name__ == "__main__":
    main()
