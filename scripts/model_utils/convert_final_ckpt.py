#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description='Convert the final checkpoint to HF format')
    parser.add_argument('checkpoint_path', type=str, help='Path to the directory containing checkpoints')
    args = parser.parse_args()
    
    try:
        # Find all directories with pattern global_step_xx
        global_step_dirs = []
        for item in os.listdir(args.checkpoint_path):
            item_path = os.path.join(args.checkpoint_path, item)
            if os.path.isdir(item_path) and item.startswith("global_step_"):
                global_step_dirs.append(item)
        
        if not global_step_dirs:
            print(f"Error: No global_step_xx directories found in {args.checkpoint_path}")
            sys.exit(1)
        
        print(f"Found {len(global_step_dirs)} checkpoint directories: {global_step_dirs}")
        
        # Process each global_step directory
        for step_dir in global_step_dirs:
            checkpoint_dir = os.path.join(args.checkpoint_path, step_dir, "actor")
            if os.path.isdir(checkpoint_dir):
                command = f"python {os.path.dirname(os.path.abspath(__file__))}/model_merger.py --local_dir {checkpoint_dir}"
                
                print(f"Running conversion command for {step_dir}: {command}")
                subprocess.run(command, shell=True, check=True)
                print(f"Conversion completed successfully for {step_dir}")
            else:
                print(f"Warning: Actor directory not found in {step_dir}, skipping")
        
        print("All conversions completed successfully")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
