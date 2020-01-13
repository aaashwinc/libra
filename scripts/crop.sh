#!/bin/sh

# while [ $begin != $end ]
# do
#   printf "%03d\n" "$begin"
#   begin=$[$begin+1]
#   # unu crop -i 000.nrrd -o miniventral/000.nrrd -min 0 100 100 150 -max 1 200 200 250
# done

cd ~/data

name=miniclearsmall

mkdir $name

for i in $( seq -w 000 100 ); do
    echo crop $i
    unu crop -i 16-05-05/$i.nrrd -o miniventral2/$i.nrrd -min 0 100 100 150 -max 1 350 350 350
    # unu crop -i $i.nrrd -o $name/$i.nrrd -min 0 0 450 0 -max 1 150 650 300
    # unu crop -i $i.nrrd -o $name/$i.nrrd -min 0 0 450 0 -max 1 150 650 300
    # unu crop -i $i.nrrd -o miniclearsmall/$i.nrrd -min 0 0 0 0 -max 1 75 582 206
done
        
# 33,  92,  165
# 75,  132, 206