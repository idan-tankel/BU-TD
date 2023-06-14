#!/bin/bash
if [ -f /etc/bashrc ]; then
      . /etc/bashrc   # --> Read /etc/bashrc, if present.
fi
# . /home/idanta/anaconda3/etc/profile.d/conda.sh  # commented out by conda initialize
# conda activate  # commented out by conda initialize

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/idanta/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/idanta/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/idanta/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/idanta/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<