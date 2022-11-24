# Use guide

## Running the shallow CNN
    chmod u+x tran_gtzan.sh;
    ./tran_gtzan.sh  

## View test result on BC4 (on BC4)
    chmod u+x tensorboard.sh;
    ./tenorboard.sh

## Data augmentation (locally)
    chmod u+x audio_augmentation.sh;  
    ./audio_augmentation.sh

*(The augmentation process will not begin
unless the previously generated data has been
cleaned (augmented(segnmented)_audio_files/"class_name"))*