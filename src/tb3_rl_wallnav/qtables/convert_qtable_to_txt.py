import numpy as np
import os
import argparse

def convert_qtable_to_txt(input_npy, output_txt):
    """Convert a Q-table .npy file to a human-readable .txt file."""
    # Get the directory of this script (qtables directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct full paths
    input_path = os.path.join(script_dir, input_npy)
    output_path = os.path.join(script_dir, output_txt)

    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist.")
        return

    try:
        # Load Q-table
        q_table = np.load(input_path, allow_pickle=True).item()

        # Sort states for consistent ordering
        states = sorted(q_table.keys())

        # Verify expected number of states (4^3 = 64)
        if len(states) != 64:
            print(f"Warning: Q-table has {len(states)} states, expected 64.")

        # Write to text file
        with open(output_path, 'w') as f:
            f.write("Q-table (State: Q-values for [Forward, Forward_Left, Forward_Right, Rotate_Left, Rotate_Right])\n")
            f.write("-" * 80 + "\n")
            for state in states:
                q_values = q_table[state]
                f.write(f"State {state}: Q-values {q_values.tolist()}\n")

        print(f"Successfully converted Q-table to '{output_path}'")

    except Exception as e:
        print(f"Error processing Q-table: {str(e)}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert a Q-table .npy file to a .txt file.")
    parser.add_argument("input_npy", help="Name of the input .npy file (e.g., qlearn_qtable.npy)")
    parser.add_argument("output_txt", help="Name of the output .txt file (e.g., qlearn_qtable.txt)")
    args = parser.parse_args()

    # Convert the Q-table
    convert_qtable_to_txt(args.input_npy, args.output_txt)

if __name__ == "__main__":
    main()