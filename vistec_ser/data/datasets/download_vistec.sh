#!/bin/bash

echo -e "+-----------------------------------+"
echo -e "| Downloading gdown...              |"
echo -e "+-----------------------------------+\n"
pip install -q -U gdown

echo -e "+-----------------------------------+"
echo -e "| Downloading dataset...            |"
echo -e "+-----------------------------------+\n"

declare download_ids
download_ids=(
  "studio1-10:1M69xuXhPE6YRFWatm0D4MiDi1blLNy0P"
  "studio11-20:1MqEestPscu2ao_jKUdM9HLva4DZFxCXe"
  "studio21-30:1lHhMEDs4YhnsGdKYBKbvidhFert_XF74"
  "studio31-40:1-AOy30Lm0yEnK_Q44QrSsgQN-XBKmfoW"
  "studio41-50:16iRYWn614AQjZoWlW9-Vc9f6TW_1Z4Ii"
  "studio51-60:1YX3Xus9hJEfbhww1mHOG_osLJho9yFBf"
  "zoom1-10:1-2QGXwfsDFfEqDl4KQ5jLPtDmbzuSc7z"
  "zoom11-20:17DXFur1ZAA7IAkX4-xa0OHyDTRa_KmZP"
  "labels:1Ym3Go5mN_5jCmvV7H3bpNnctk_tRqhNb"
)

for item in ${download_ids[@]}; do
  f="${item%%:*}"
  id="${item##*:}"
  if [ ! -f ${f}.zip ]; then
    echo ">downloading $f.zip ..."
    gdown https://drive.google.com/uc?id=${id} || exit 1;
  else
    echo "${f}.zip existed, skipping..."
  fi
done
echo "Finished Downloading Dataset\n"

echo -e "+-----------------------------------+"
echo -e "| Extracting dataset...             |"
echo -e "+-----------------------------------+\n"

for f in $(find . -name "*.zip"); do
  echo ">unzipping ${f}..."
  unzip -q $f
done

mkdir -p studio zoom labels raw
mv studio0* studio
mv zoom0* zoom
mv *.json labels
mv *.zip raw

echo "Finished Extracting Dataset"

echo -e "+-----------------------------------+"
echo -e "| Formatting labels...              |"
echo -e "+-----------------------------------+\n"

for j in labels/*.json; do
  echo ">formatting $j ..."
  csv_path=$(basename $j)
  python generate_csv.py --csv-path ${csv_path/json/csv} --threshold 0.7 --json-path $j
done
cat *.csv | sort -u > labels.csv
mv studio*.csv zoom*.csv labels