#!/bin/sh

# while [ $begin != $end ]
# do
#   printf "%03d\n" "$begin"
#   begin=$[$begin+1]
#   # unu crop -i 000.nrrd -o miniventral/000.nrrd -min 0 100 100 150 -max 1 200 200 250
# done

cd ~/data

for i in $( seq -w 000 020 ); do
    echo crop $i
    unu crop -i $i.nrrd -o miniventral/$i.nrrd -min 0 100 100 150 -max 1 250 250 350
done
        