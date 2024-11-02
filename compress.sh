tar --use-compress-program='pigz --fast --recursive | pv' \
    -cf submission.tar.gz \
    -C ./exp007 . \
    # -C /media/ssd2/kaggle2/LLM20Questions/input deepseek-math


