

start-session() {
  tmux new -s $1
}

attach-session() {
  tmux attach -t $1
}

delete-session() {
  tmux kill-session -t $1
}

# Get out of session in terminal 
# Ctrl + B, D


# Parse arguments from command line
if [ "$1" = "start" ]; then
    start-session "$2"
elif [ "$1" = "attach" ]; then
    attach-session "$2"
elif [ "$1" = "delete" ]; then
    delete-session "$2"
else
    echo "Invalid command"
fi