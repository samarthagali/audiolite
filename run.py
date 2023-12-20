import sys
import subprocess
# Check if two arguments are provided
if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} argument1 argument2")
    sys.exit(1)

# Assign command-line arguments to variables
arg1 = sys.argv[1]
arg2 = sys.argv[2]
command = ["python", "encode.py", arg1, arg2]

# Run the command
process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
print(process.stdout,process.stderr )

command = ["python", "decode.py", arg1]

# Run the command
process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
print(process.stdout,process.stderr )

