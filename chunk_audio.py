# Import necessary libraries
import sys
import argparse
from pydub import AudioSegment
from pydub import silence

parser = argparse.ArgumentParser(description='Split an input .wav file into chunks')
parser.add_argument('input_file', type=str, help='Path to the input .wav file')
parser.add_argument('output_path', type=str, help='Output path for the chunks')
parser.add_argument('--chunk_length', type=int, default=8000, help='Output chunk size in milliseconds')
parser.add_argument('--overlap', type=int, default=0, help='Overlap between consecutive chunks in milliseconds')

args = parser.parse_args()

input_file = args.input_file
output_path = args.output_path
chunk_length = args.chunk_length
overlap = args.overlap

# Function to check for silent chunks
def is_silent(chunk, chunk_length):
    # if at least half of chunk is silence, mark as silent
    silences = silence.detect_silence(chunk,
									  min_silence_len=int(chunk_length/2),
									  silence_thresh=-64)
    if silences:
        return True
    else:
        return False

# Load file and set directories
album_audio = AudioSegment.from_file(input_file)
album_file = input_file.split('/')[-1]
album_name = album_file.replace('.wav', '')

# Get length of album
album_length = len(album_audio)

# Initialize start and end seconds to 0
start = 0
end = 0

# Iterate from 0 to end of the file,
# with increment = chunk_length
cnt = 1
flag = 0 # use to break loop once reach end of album
for i in range(0, 8 * album_length, chunk_length):

	# At first, start is 0, end is the chunk_length
	# Else, start=prvs end-overlap, end=start+chunk_length
	if i == 0:
		start = 0
		end = chunk_length
	else:
		start = end - overlap
		end = start + chunk_length

	# Set flag to 1 if endtime exceeds length of file
	if end >= album_length:
		end = album_length
		flag = 1

	# Storing audio file from the defined start to end
	chunk = album_audio[start:end]
	if flag == 0 and not is_silent(chunk, chunk_length):
		filename = album_name + f'_chunk_{cnt}.wav'
		chunk.export(output_path + filename, format ="wav")
		print("Processing chunk " + str(cnt) + ". Start = "
				+ str(start) + " end = " + str(end))

	cnt = cnt + 1
