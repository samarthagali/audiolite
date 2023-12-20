if [ "$#" -ne 2 ]; then
    echo "Usage: $0 models_dir  audio_file_path"
    exit 1
fi

python3 encoder.py $1 $2
python3 decoder.py $1


