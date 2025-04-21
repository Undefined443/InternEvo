# for i in $(seq -w 0 29); do
#     TMUX="tmux neww -t the_pile -n $i"
#     ENV="source $HOME/.zshrc; conda activate InternEvo"
#     COMMAND="$ENV; python /data/ubuntu/InternEvo/tools/the_pile_tokenizer_serial.py /data/lx/dataset/pile-uncopyrighted/train/$i.jsonl.zst /data/ubuntu/InternEvo/data/the_pile/train/roberta/en --model=FacebookAI/roberta-base; sleep 5"
#     RUN="$TMUX \"$COMMAND\""
#     eval $RUN
# done

ENV="source $HOME/.zshrc; conda activate InternEvo"
SCRIPT="/data/ubuntu/InternEvo/tools/the_pile_tokenizer_serial.py"
INPUT_DIR="/data/lx/dataset/minipile/data"
OUTPUT_DIR="/data/ubuntu/InternEvo/data/minipile/train/roberta/en"
COMMAND="ls $INPUT_DIR | grep train | xargs -I {} -n 1 tmux neww -t minipile \"$ENV; python $SCRIPT $INPUT_DIR/{} $OUTPUT_DIR --model=FacebookAI/roberta-base\""
echo $COMMAND
eval $COMMAND